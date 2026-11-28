import ctypes
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
import sys

from PySide6.QtCore import QIODevice, QSaveFile, QStandardPaths, QEasingCurve, QPropertyAnimation, QSettings, QThread, QUrl, Qt, Signal
from PySide6.QtGui import QCursor, QDesktopServices, QIcon, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget,
    QMainWindow, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QStyle, QStyleOptionSlider, QVBoxLayout, QWidget,
)

from apollo_assets import APOLLO_URL, apollo_ready
from sidon_assets import sidon_ready
from studio_formats import DEFAULT_OUTPUT_FORMAT, OUTPUT_FORMATS

ROOT = Path(__file__).resolve().parent

WEIGHTS = "https://drive.google.com/drive/folders/1IIYTf2zbJWzelu4IftKD6ooHloJ8mnZF"
SIDON_URL = "https://huggingface.co/soniqo/Sidon-ONNX"
PROFILES = ("8kto16k", "8kto48k", "12kto48k", "16kto48k", "24kto48k", "4kto16k", "2kto16k")

STYLE = """
QWidget { background: #11131d; color: #eef0fa; font: 14px 'Segoe UI'; }
QFrame#card { background: #191c29; border: 1px solid #2d3044; border-radius: 16px; }
QFrame#card QLabel, QFrame#card QWidget#controls, QSlider { background: transparent; }
QLabel#title { font-size: 40px; font-weight: 700; }
QLabel#compactTitle { font-size: 34px; font-weight: 700; }
QLabel#brand { color: #b6a5ff; font-size: 13px; font-weight: 700; }
QLabel#muted { color: #aeb5cb; }
QLabel#section { font-size: 19px; font-weight: 650; }
QPushButton, QComboBox {
  background: #24283a; border: 1px solid #373c54; border-radius: 9px; padding: 7px 12px; min-height: 22px;
}
QComboBox { border-radius: 14px; padding-right: 32px; }
QComboBox::drop-down {
  subcontrol-origin: padding; subcontrol-position: top right;
  width: 28px; border: none; background: transparent;
}
QComboBox::down-arrow { image: url("__CHEVRON__"); width: 12px; height: 8px; }
QComboBox QAbstractItemView {
  background: #191c29; border: 1px solid #373c54; border-radius: 10px;
  padding: 4px; selection-background-color: #3b345c; selection-color: #f4edff;
  outline: none;
}
QAbstractItemView#outputFormats QScrollBar:vertical {
  background: #191c29; width: 9px; margin: 3px 0; border: none;
}
QAbstractItemView#outputFormats QScrollBar::handle:vertical {
  background: #606482; border-radius: 4px; min-height: 24px;
}
QAbstractItemView#outputFormats QScrollBar::handle:vertical:hover { background: #a592ff; }
QAbstractItemView#outputFormats QScrollBar::add-line:vertical,
QAbstractItemView#outputFormats QScrollBar::sub-line:vertical { height: 0; border: none; }
QAbstractItemView#outputFormats QScrollBar::add-page:vertical,
QAbstractItemView#outputFormats QScrollBar::sub-page:vertical { background: transparent; }
QPushButton:hover, QComboBox:hover { border-color: #a592ff; background: #313450; }
QPushButton:pressed { background: #414364; }
QPushButton:disabled { color: #858aa0; background: #212433; border-color: #303347; }
QPushButton#primary { background: #b6a5ff; color: #161229; font-weight: 700; border-color: #b6a5ff; }
QPushButton#primary:hover { background: #cebdff; }
QPushButton#primary:disabled { background: #333048; color: #a6a0b9; border-color: #393550; }
QListWidget { background: #141622; border: 1px solid #32354c; border-radius: 10px; padding: 5px; outline: none; }
QListWidget::item { padding: 9px; border-radius: 6px; }
QListWidget::item:selected { background: #3b345c; color: #f4edff; }
QProgressBar { background: #25283b; border: 0; border-radius: 4px; }
QProgressBar::chunk { background: #b6a5ff; border-radius: 4px; }
QSlider::groove:horizontal { height: 4px; background: #383b53; border-radius: 2px; }
QSlider::handle:horizontal { background: #b6a5ff; width: 14px; margin: -5px 0; border-radius: 7px; }
"""


STYLE = STYLE.replace("__CHEVRON__", (ROOT / "chevron.png").as_posix())


class SeekSlider(QSlider):
    def slider_geometry(self):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        style = self.style()
        control = QStyle.ComplexControl.CC_Slider
        groove = style.subControlRect(control, option, QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(control, option, QStyle.SubControl.SC_SliderHandle, self)
        return groove, handle, option.upsideDown

    def seek_at(self, x):
        groove, handle, upside_down = self.slider_geometry()
        span = max(1, groove.width() - handle.width())
        offset = x - groove.left() - self.drag_offset
        self.setSliderPosition(QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), max(0, min(offset, span)), span, upside_down
        ))

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        _, handle, _ = self.slider_geometry()
        point = event.position().toPoint()
        self.drag_offset = point.x() - handle.left() if handle.contains(point) else handle.width() // 2
        self.setSliderDown(True)
        self.seek_at(point.x())
        event.accept()

    def mouseMoveEvent(self, event):
        if self.isSliderDown() and event.buttons() & Qt.MouseButton.LeftButton:
            self.seek_at(round(event.position().x()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.isSliderDown() and event.button() == Qt.MouseButton.LeftButton:
            self.seek_at(round(event.position().x()))
            self.setSliderDown(False)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class Batch(QThread):
    message = Signal(str)
    progress = Signal(int)
    exported = Signal(object, object)
    report = Signal(str)

    def __init__(self, files, folder, profile, checkpoint, device, output_format, parent):
        super().__init__(parent)
        self.files, self.folder, self.profile = files, folder, profile
        self.checkpoint, self.device = checkpoint, device
        self.extension, self.encoder_options = OUTPUT_FORMATS[output_format]

    def run(self):
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            if self.profile == "sidon":
                from studio_sidon import Sidon

                engine = Sidon(self.checkpoint, self.device)
                process = engine.enhance
                device = engine.device
            elif self.profile == "apollo":
                from studio_apollo import Apollo

                engine = Apollo(self.checkpoint, self.device)
                process = engine.enhance
                device = engine.device
            else:
                from studio_engine import enhance, load_model

                model, config, device = load_model(
                    ROOT / "configs" / f"config_{self.profile}.json", self.checkpoint, self.device
                )
                process = partial(enhance, model=model, h=config, device=device)
        except Exception as error:
            self.report.emit(f"Could not start: {error}")
            return

        completed = 0
        errors = []
        total = len(self.files)
        for index, source in enumerate(self.files):
            if self.isInterruptionRequested():
                break
            self.message.emit(f"Processing {index + 1}/{total} · {source.name} · {str(device).upper()}")
            try:
                result_dir = Path(mkdtemp(dir=self.folder))
                destination = result_dir / f"{source.stem} ilaria audio upscaler{self.extension}"

                def update(fraction):
                    self.progress.emit(round(1000 * (index + fraction) / total))

                if not process(source, destination, progress=update, cancelled=self.isInterruptionRequested,
                               encoder_options=self.encoder_options):
                    break
                completed += 1
                self.exported.emit(source, destination)
            except Exception as error:
                errors.append(f"{source.name}: {error}")
            self.progress.emit(round(1000 * (index + 1) / total))

        summary = f"Converted {completed}/{total} audio files."
        if completed:
            summary += " Use Save result to keep a copy."
        if self.isInterruptionRequested():
            summary = f"Cancelled. {summary}"
        if errors:
            summary += f" {len(errors)} failed. {errors[0]}"
        self.report.emit(summary)



class GpuScan(QThread):
    detected = Signal(list, str)

    def run(self):
        try:
            import torch

            devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            self.detected.emit(devices, "" if devices else "No CUDA GPUs detected.")
        except Exception as error:
            self.detected.emit([], f"Could not detect CUDA GPUs: {error}")


class CheckpointDownload(QThread):
    message = Signal(str)
    progress = Signal(int)
    report = Signal(str)

    def __init__(self, profile, destination, parent):
        super().__init__(parent)
        self.profile = profile
        self.destination = destination

    def run(self):
        from studio_download import download_checkpoint

        try:
            if self.profile == "sidon":
                from sidon_assets import download_sidon

                saved = download_sidon(self.destination, self.update, self.isInterruptionRequested)
            elif self.profile == "apollo":
                from apollo_assets import download_apollo

                saved = download_apollo(self.destination, progress=self.update,
                                        cancelled=self.isInterruptionRequested)
            else:
                saved = download_checkpoint(self.profile, self.destination, self.update,
                                            self.isInterruptionRequested)
            self.report.emit(f"Checkpoint ready · {self.destination.name}" if saved else
                             "Download cancelled.")
        except Exception as error:
            if self.isInterruptionRequested():
                self.report.emit("Download cancelled.")
            else:
                self.report.emit(f"Download failed: {error}")

    def update(self, received, total):
        megabytes = received / 1048576
        if total:
            self.progress.emit(min(990, round(received * 1000 / total)))
            self.message.emit(f"Downloading checkpoint · {megabytes:.1f} / {total / 1048576:.1f} MB")
        else:
            self.progress.emit(-1)
            self.message.emit(f"Downloading checkpoint · {megabytes:.1f} MB")


def label(text, style=""):
    widget = QLabel(text)
    widget.setObjectName(style)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    return widget


def card(heading, compact=False):
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    vertical_margin = 8 if compact else 16
    layout.setContentsMargins(18, vertical_margin, 18, vertical_margin)
    layout.setSpacing(4 if compact else 10)
    layout.addWidget(label(heading, "section"))
    return frame, layout


class Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ilaria Audio Upscaler")
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        available = screen.availableGeometry()
        compact = available.height() < 850
        self.setMinimumSize(min(825, available.width() - 24), min(480, available.height() - 56))
        self.setAcceptDrops(True)
        self.settings = QSettings("AP-BWE Studio", "Studio")
        self.results_directory = TemporaryDirectory(prefix="ilaria-audio-", ignore_cleanup_errors=True)
        self.engine_results = {"ap-bwe": {}, "sidon": {}, "apollo": {}}
        self.preview_source = None
        self.worker = None
        self.gpu_scan = None
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.75)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio)

        panel = QWidget()
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        self.setCentralWidget(scroll)
        layout = QVBoxLayout(panel)
        vertical_margin = 6 if compact else 22
        layout.setContentsMargins(24, vertical_margin, 24, vertical_margin)
        layout.setSpacing(6 if compact else 13)
        self.heading = label("Ilaria Audio Upscaler", "compactTitle" if compact else "title")
        self.engine_description = label("", "muted")
        header = QHBoxLayout()
        header.setSpacing(12)
        self.header_image = QLabel()
        self.header_image.setAccessibleName("Ilaria")
        portrait_size = 42 if compact else 50
        self.header_image.setFixedSize(portrait_size, portrait_size)
        portrait = QPixmap(str(ROOT / "icon.png"))
        self.header_image.setPixmap(portrait.scaled(
            portrait_size, portrait_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        header.addWidget(self.header_image, alignment=Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.heading)
        header.addStretch()
        layout.addLayout(header)
        layout.addWidget(self.engine_description)

        work_area = QHBoxLayout()
        work_area.setSpacing(13)
        layout.addLayout(work_area, 1)
        queue_frame, queue = card("Input queue", compact)
        work_area.addWidget(queue_frame, 3)
        drop_hint = label("Drop audio files or videos with an audio track here.", "muted")
        drop_hint.setWordWrap(True)
        queue.addWidget(drop_hint)
        self.files = QListWidget()
        self.files.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.files.setMinimumHeight(86)
        self.files.currentItemChanged.connect(self.refresh)
        queue.addWidget(self.files, 1)
        queue_actions = QHBoxLayout()
        queue.addLayout(queue_actions)
        self.add_button = self.button(queue_actions, "Add audio", self.choose_audio)
        self.remove_button = self.button(queue_actions, "Remove selected", self.remove)
        queue_actions.addStretch()

        config_frame, config = card("Processing", compact)
        work_area.addWidget(config_frame, 2)
        selectors = QGridLayout()
        selectors.setHorizontalSpacing(10)
        selectors.setVerticalSpacing(6)
        selectors.setColumnStretch(0, 1)
        selectors.setColumnStretch(1, 1)
        config.addLayout(selectors)
        self.selectors = selectors
        selectors.addWidget(label("Engine", "muted"), 0, 0)
        self.engine = QComboBox()
        self.engine.addItem("Sidon", "sidon")
        self.engine.addItem("AP-BWE", "ap-bwe")
        self.engine.addItem("Apollo", "apollo")
        self.engine.setCurrentIndex(self.engine.findData("sidon"))
        selectors.addWidget(self.engine, 1, 0)
        self.engine.currentIndexChanged.connect(self.engine_changed)

        self.profile_label = label("Model profile", "muted")
        selectors.addWidget(self.profile_label, 0, 1)
        self.profile = QComboBox()
        for profile in PROFILES:
            source, target = profile.split("kto")
            self.profile.addItem(f"{source} → {target[:-1]} kHz", profile)
        self.profile.setCurrentIndex(self.profile.findData("16kto48k"))
        selectors.addWidget(self.profile, 1, 1)
        self.profile.currentIndexChanged.connect(self.refresh)
        self.device_row = QWidget()
        self.device_row.setObjectName("controls")
        self.device_row_layout = QHBoxLayout(self.device_row)
        self.device_row_layout.setContentsMargins(0, 0, 0, 0)
        self.device_row_layout.addStretch()
        self.device_row_layout.addStretch()
        config.addWidget(self.device_row)
        self.device_label = label("Compute device", "muted")
        self.device = QComboBox()
        self.device.addItems(["Auto", "CPU", "CUDA"])
        self.device.setMinimumWidth(136)
        self.place_device_selector()
        self.gpu_row = QWidget()
        self.gpu_row.setObjectName("controls")
        gpu_controls = QHBoxLayout(self.gpu_row)
        gpu_controls.setContentsMargins(0, 0, 0, 0)
        gpu_controls.addWidget(label("GPU", "muted"))
        self.gpu = QComboBox()
        self.gpu.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.gpu.setMinimumContentsLength(12)
        gpu_controls.addWidget(self.gpu, 1)
        config.addWidget(self.gpu_row)
        self.gpu_row.hide()
        self.device.currentTextChanged.connect(self.device_changed)
        self.weights_status = label("", "muted")
        self.weights_status.setWordWrap(True)
        config.addWidget(self.weights_status)
        weight_actions = QHBoxLayout()
        config.addLayout(weight_actions)
        self.weights_button = self.button(weight_actions, "Choose checkpoint…", self.choose_checkpoint)
        self.button(weight_actions, "Official weights ↗",
                    self.open_model_page)
        self.download_button = self.button(config, "Download checkpoint", self.download_action)
        self.download_bar = QProgressBar()
        self.download_bar.setFixedHeight(7)
        self.download_bar.setTextVisible(False)
        self.download_bar.hide()
        config.addWidget(self.download_bar)
        config.addStretch()

        preview_frame, preview = card("Listen", compact)
        layout.addWidget(preview_frame)
        controls = QHBoxLayout()
        preview.addLayout(controls)
        self.original = self.button(controls, "Play original", lambda: self.listen(self.selection()))
        self.result = self.button(controls, "Play result",
                                  lambda: self.listen(self.outputs.get(self.selection())))
        self.save_result_button = self.button(controls, "Save result…", self.save_result)
        controls.addStretch()
        controls.addWidget(label("Speed", "muted"))
        self.speed = QComboBox()
        for speed in ("0.25", "0.50", "0.75", "1", "1.5", "2", "4", "8", "16", "32"):
            self.speed.addItem(f"{speed}×", float(speed))
        self.speed.setCurrentIndex(self.speed.findData(1.0))
        self.speed.currentIndexChanged.connect(
            lambda _: self.player.setPlaybackRate(self.speed.currentData())
        )
        controls.addWidget(self.speed)

        transport = QHBoxLayout()
        preview.addLayout(transport)
        self.play = self.button(transport, "Play", self.play_preview)
        self.pause = self.button(transport, "Pause", self.player.pause)
        self.stop = self.button(transport, "Stop", self.player.stop)
        self.seek = SeekSlider(Qt.Orientation.Horizontal)
        self.seek.setEnabled(False)
        self.seek.sliderMoved.connect(self.player.setPosition)
        transport.addWidget(self.seek, 1)
        self.time = label("0:00 / 0:00", "muted")
        transport.addWidget(self.time)
        self.now_playing = label("Select a file to preview it.", "muted")
        preview.addWidget(self.now_playing)
        self.player.positionChanged.connect(self.sync_player)
        self.player.durationChanged.connect(self.sync_player)
        self.player.playbackStateChanged.connect(self.sync_player)
        self.player.errorOccurred.connect(lambda _, message: self.now_playing.setText(message))

        export = QHBoxLayout()
        layout.addLayout(export)
        export.addStretch()
        export.addWidget(label("Output format", "muted"))
        self.output_format = QComboBox()
        self.output_format.addItems(list(OUTPUT_FORMATS))
        self.output_format.view().setObjectName("outputFormats")
        saved_format = self.settings.value("output_format", DEFAULT_OUTPUT_FORMAT)
        self.output_format.setCurrentText(saved_format if saved_format in OUTPUT_FORMATS else DEFAULT_OUTPUT_FORMAT)
        self.output_format.currentTextChanged.connect(
            lambda value: self.settings.setValue("output_format", value)
        )
        export.addWidget(self.output_format)
        self.cancel = self.button(export, "Cancel", self.cancel_batch)
        self.start = self.button(export, "Enhance audio", self.start_batch)
        self.start.setObjectName("primary")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(7)
        layout.addWidget(self.bar)
        self.animation = QPropertyAnimation(self.bar, b"value", self)
        self.animation.setDuration(220)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.status = label("Add audio to begin.", "muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.refresh()

        # Reserve space for the native title bar and resize borders.
        width = min(1050, available.width() - 24)
        height = min(max(735, panel.sizeHint().height() + 24), available.height() - 40)
        self.resize(width, height)
        self.move(available.x() + (available.width() - width) // 2,
                  available.y() + max(4, (available.height() - height - 36) // 2))

    @staticmethod
    def button(layout, title, action):
        widget = QPushButton(title)
        widget.clicked.connect(action)
        widget.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(widget)
        return widget

    @property
    def outputs(self):
        return self.engine_results[self.engine.currentData()]

    def engine_changed(self):
        self.clear_preview()
        self.place_device_selector()
        self.refresh()

    def clear_preview(self):
        self.player.stop()
        self.player.setSource(QUrl())
        self.preview_source = None
        self.now_playing.setText("Select a file to preview it.")

    def place_device_selector(self):
        fixed_profile = self.engine.currentData() != "ap-bwe"
        if fixed_profile:
            self.selectors.addWidget(self.device_label, 0, 1)
            self.selectors.addWidget(self.device, 1, 1)
        else:
            self.device_row_layout.insertWidget(1, self.device_label)
            self.device_row_layout.insertWidget(2, self.device)
        self.device_row.setVisible(not fixed_profile)

    def open_model_page(self):
        urls = {"sidon": SIDON_URL, "apollo": APOLLO_URL, "ap-bwe": WEIGHTS}
        QDesktopServices.openUrl(QUrl(urls[self.engine.currentData()]))

    def model_profile(self):
        engine = self.engine.currentData()
        return self.profile.currentData() if engine == "ap-bwe" else engine

    def checkpoint_destination(self):
        profile = self.model_profile()
        folder = {"sidon": "Sidon-INT8", "apollo": "Apollo"}.get(profile, f"g_{profile}")
        return ROOT / "checkpoints" / folder

    def checkpoint(self):
        engine = self.engine.currentData()
        if engine != "ap-bwe":
            folder = self.checkpoint_destination()
            ready = sidon_ready if engine == "sidon" else apollo_ready
            return folder if ready(folder) else None
        profile = self.profile.currentData()
        saved = self.settings.value(f"checkpoint/{profile}", "")
        if saved and Path(saved).is_file():
            return Path(saved)
        for candidate in (ROOT / "checkpoints" / f"g_{profile}",
                          ROOT / "checkpoints" / profile / f"g_{profile}"):
            if candidate.is_file():
                return candidate
        return None

    def device_changed(self, device):
        self.gpu_row.setVisible(device == "CUDA")
        if device == "CUDA" and not self.gpu.count():
            self.gpu.addItem("Detecting CUDA GPUs…")
            self.gpu_scan = GpuScan(self)
            self.gpu_scan.detected.connect(self.show_cuda_devices)
            self.gpu_scan.finished.connect(self.finish_gpu_scan)
            self.gpu_scan.start()
        self.refresh()

    def show_cuda_devices(self, devices, error):
        self.gpu.clear()
        self.gpu.setToolTip(error)
        for index, name in enumerate(devices):
            self.gpu.addItem(f"GPU {index} · {name}", f"cuda:{index}")
        if not devices:
            self.gpu.addItem("No CUDA GPUs available")
        self.refresh()

    def finish_gpu_scan(self):
        self.gpu_scan.deleteLater()
        self.gpu_scan = None

    def selection(self):
        item = self.files.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def refresh(self):
        checkpoint = self.checkpoint()
        busy = self.worker is not None
        selected = self.selection()
        engine = self.engine.currentData()
        ap_bwe = engine == "ap-bwe"
        descriptions = {
            "sidon": "Restore speech with background noise or reverb.",
            "ap-bwe": "Extend the high frequencies of speech with AP-BWE.",
            "apollo": "Restore music degraded by lossy compression.",
        }
        self.engine_description.setText(descriptions[engine])
        self.profile_label.setVisible(ap_bwe)
        self.profile.setVisible(ap_bwe)
        self.weights_button.setVisible(ap_bwe)
        downloading = isinstance(self.worker, CheckpointDownload)
        if not downloading:
            if engine == "sidon":
                status = ("Sidon INT8 ready · speech restoration to 48 kHz" if checkpoint else
                          "Sidon INT8 required · 286 MiB download")
            elif engine == "apollo":
                status = ("Apollo ready · music restoration at 44.1 kHz" if checkpoint else
                          "Apollo required · 63.5 MiB download")
            else:
                status = (f"Checkpoint available · {checkpoint.name}" if checkpoint else
                          "Checkpoint required for this profile.")
            self.weights_status.setText(status)
        device_ready = self.device.currentText() != "CUDA" or self.gpu.currentData() is not None
        self.start.setEnabled(bool(self.files.count() and checkpoint and not busy and device_ready))
        self.gpu.setEnabled(not busy and self.gpu.currentData() is not None)
        self.cancel.setEnabled(busy and not self.worker.isInterruptionRequested())
        if downloading:
            cancelling = self.worker.isInterruptionRequested()
            self.download_button.setText("Cancelling download…" if cancelling else "Cancel download")
            self.download_button.setEnabled(not cancelling)
        else:
            model_name = {"sidon": "Sidon INT8", "apollo": "Apollo"}.get(engine, "checkpoint")
            ready_text = f"{model_name[0].upper()}{model_name[1:]} already available"
            download_text = f"Download {model_name}"
            self.download_button.setText(ready_text if checkpoint else download_text)
            self.download_button.setEnabled(not busy and checkpoint is None)
        self.remove_button.setEnabled(selected is not None and not busy)
        self.original.setEnabled(selected is not None)
        result = self.outputs.get(selected)
        has_result = result is not None and result.is_file()
        self.result.setEnabled(has_result)
        self.save_result_button.setEnabled(has_result)
        self.sync_player()
        for widget in (self.add_button, self.engine, self.profile, self.device,
                       self.weights_button, self.output_format):
            widget.setEnabled(not busy)

    def add_files(self, paths):
        if self.worker:
            return
        existing = {self.files.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.files.count())}
        for path in map(Path, paths):
            path = path.resolve()
            if path in existing or not path.is_file():
                continue
            self.files.addItem(path.name)
            item = self.files.item(self.files.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(str(path))
            existing.add(path)
        if self.files.count() and not self.files.currentItem():
            self.files.setCurrentRow(0)
        self.status.setText(f"{self.files.count()} audio files in queue.")
        self.refresh()

    def dragEnterEvent(self, event):
        if self.worker is None and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.add_files([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def choose_audio(self):
        downloads = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add audio", downloads, "All files (*)"
        )
        if paths:
            self.add_files(paths)

    def remove(self):
        source = self.selection()
        if source is None:
            return
        if source == self.preview_source:
            self.clear_preview()
        for results in self.engine_results.values():
            results.pop(source, None)
        self.files.takeItem(self.files.currentRow())
        self.refresh()

    def choose_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose AP-BWE generator checkpoint",
                                             str(ROOT / "checkpoints"))
        if path:
            self.settings.setValue(f"checkpoint/{self.profile.currentData()}", path)
            self.refresh()

    def listen(self, path):
        if path:
            self.preview_source = self.selection()
            self.player.setSource(QUrl.fromLocalFile(str(path)))
            self.now_playing.setText(f"Playing · {path.name}")
            self.player.play()

    def save_result(self):
        source = self.outputs.get(self.selection())
        if source is None:
            return
        downloads = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        suggested = Path(downloads) / source.name if downloads else source
        path, _ = QFileDialog.getSaveFileName(
            self, "Save converted audio", str(suggested), "All files (*)"
        )
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != source.suffix.lower():
            destination = destination.with_name(destination.name + source.suffix)
            if destination.exists():
                self.status.setText(f"Choose a different name. {destination.name} already exists.")
                return
        if destination.resolve() == source.resolve():
            self.status.setText(f"Already saved · {source}")
            return

        output = QSaveFile(str(destination))
        try:
            with source.open("rb") as audio:
                if not output.open(QIODevice.OpenModeFlag.WriteOnly):
                    raise OSError(output.errorString())
                while block := audio.read(1024 * 1024):
                    if output.write(block) != len(block):
                        raise OSError(output.errorString())
                if not output.commit():
                    raise OSError(output.errorString())
        except OSError as error:
            output.cancelWriting()
            self.status.setText(f"Could not save result: {error}")
        else:
            self.status.setText(f"Saved result · {destination}")

    def play_preview(self):
        if self.player.source().isEmpty():
            self.listen(self.selection())
        else:
            self.player.play()

    def sync_player(self, *_):
        duration = self.player.duration()
        position = self.player.position()
        self.seek.setEnabled(duration > 0)
        has_source = not self.player.source().isEmpty()
        state = self.player.playbackState()
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play.setEnabled(not playing and (has_source or self.selection() is not None))
        self.pause.setEnabled(playing)
        self.stop.setEnabled(has_source and state != QMediaPlayer.PlaybackState.StoppedState)
        if not self.seek.isSliderDown():
            self.seek.setRange(0, duration)
            self.seek.setValue(position)

        def timestamp(milliseconds):
            minutes, seconds = divmod(milliseconds // 1000, 60)
            return f"{minutes}:{seconds:02d}"

        self.time.setText(f"{timestamp(position)} / {timestamp(duration)}")

    def download_action(self):
        if isinstance(self.worker, CheckpointDownload):
            self.cancel_batch()
        else:
            self.start_download()

    def start_download(self):
        if self.worker is not None or self.checkpoint():
            return
        profile = self.model_profile()
        destination = self.checkpoint_destination()
        self.download_bar.setRange(0, 0)
        self.download_bar.show()
        self.weights_status.setText("Connecting to Hugging Face…" if profile in ("sidon", "apollo") else "Connecting to Google Drive…")
        self.status.setText("Downloading checkpoint…")
        self.worker = CheckpointDownload(profile, destination, self)
        self.worker.message.connect(self.download_message)
        self.worker.progress.connect(self.download_progress)
        self.worker.report.connect(self.download_report)
        self.worker.finished.connect(self.download_finished)
        self.refresh()
        self.worker.start()

    def download_message(self, message):
        if self.worker and not self.worker.isInterruptionRequested():
            self.weights_status.setText(message)

    def download_progress(self, value):
        if value >= 0:
            self.download_bar.setRange(0, 1000)
            self.download_bar.setValue(value)

    def download_report(self, message):
        self.status.setText(message)
        self.last_download_report = message

    def download_finished(self):
        self.download_bar.hide()
        self.batch_finished()
        if not self.checkpoint():
            self.weights_status.setText(self.last_download_report)

    def start_batch(self):
        checkpoint = self.checkpoint()
        if not checkpoint:
            return
        files = [self.files.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.files.count())]
        self.animation.stop()
        self.bar.setValue(0)
        self.status.setText(f"Loading the {self.engine.currentText()} model…")
        device = self.gpu.currentData() if self.device.currentText() == "CUDA" else self.device.currentText()
        if device is None:
            return
        profile = self.model_profile()
        self.worker = Batch(files, Path(self.results_directory.name), profile, checkpoint,
                            device, self.output_format.currentText(), self)
        self.worker.message.connect(self.status.setText)
        self.worker.progress.connect(self.animate_progress)
        self.worker.exported.connect(self.register_output)
        self.worker.report.connect(self.status.setText)
        self.worker.finished.connect(self.batch_finished)
        self.refresh()
        self.worker.start()

    def animate_progress(self, value):
        if value < 0:
            self.animation.stop()
            self.bar.setRange(0, 0)
            return
        if self.bar.maximum() == 0:
            self.bar.setRange(0, 1000)
            self.bar.setValue(0)

        self.animation.stop()
        self.animation.setStartValue(self.bar.value())
        self.animation.setEndValue(value)
        self.animation.start()

    def register_output(self, source, destination):
        self.outputs[source] = destination
        self.refresh()

    def cancel_batch(self):
        if self.worker:
            self.worker.requestInterruption()
            self.cancel.setEnabled(False)
            if isinstance(self.worker, CheckpointDownload):
                self.weights_status.setText("Cancelling download…")
                self.refresh()
            else:
                self.status.setText("Cancelling after the current audio segment…")

    def batch_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.refresh()

    def closeEvent(self, event):
        if self.gpu_scan is not None:
            event.ignore()
        elif self.worker:
            self.cancel_batch()
            event.ignore()
        else:
            self.player.setSource(QUrl())
            self.results_directory.cleanup()
            super().closeEvent(event)


def main():
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("APBWE.Studio")
    app = QApplication(sys.argv)
    app.setApplicationDisplayName("Ilaria Audio Upscaler")
    app.setWindowIcon(QIcon(str(ROOT / "icon.png")))
    app.setStyleSheet(STYLE)
    studio = Studio()
    studio.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
