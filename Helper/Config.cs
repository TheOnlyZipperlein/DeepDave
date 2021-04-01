using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace DeepDave.Helper {
    public class Config { 
        public static float learningRate { get; set; } = 0.001f;
        public static bool learningEnabled { get; set; } = false;
        public static bool KeepDataForNewEpoches { get; set; } = true;
        public static bool DebuggingToggle { get; set; } = false;
        public static Size inputSize, outputSize;
        public static Dave currentDave;
        
    }
}
