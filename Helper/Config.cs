using System.Drawing;

namespace DeepDave.Helper {
    public class Config {
        public static float learningRate { get; set; } = 0.01f;
        public static bool learningEnabled { get; set; } = false;
        public static bool KeepDataForNewEpoches { get; set; } = true;
        public static bool DebuggingToggle { get; set; } = false;
        public static Size inputSize, outputSize;
        public static Dave currentDave;

    }
}
