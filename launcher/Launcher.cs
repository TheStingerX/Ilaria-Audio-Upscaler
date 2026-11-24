using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class Launcher
{
    [STAThread]
    private static int Main()
    {
        string folder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "App");
        string interpreter = Path.Combine(folder, @".venv\Scripts\pythonw.exe");
        string script = Path.Combine(folder, "studio.py");

        try
        {
            if (!File.Exists(interpreter) || !File.Exists(script))
                throw new FileNotFoundException("Keep Ilaria Audio Upcaler.exe in its installation folder. If the Python environment is missing, run Setup.cmd inside the App folder.");

            var options = new ProcessStartInfo(interpreter, "\"" + script + "\"");
            options.WorkingDirectory = folder;
            options.UseShellExecute = false;
            using (Process app = Process.Start(options))
            {
                app.WaitForExit();
                if (app.ExitCode != 0)
                    throw new InvalidOperationException("The studio closed with exit code " + app.ExitCode + ".");
            }
            return 0;
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "Ilaria Audio Upcaler", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}


