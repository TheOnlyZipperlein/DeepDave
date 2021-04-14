
using System;
using System.Collections.Generic;
using System.Text;
 
namespace DeepDave.Layer {
    public partial class Layer2D {
        public class ActivationFunctions {
            public const string FastSigmoidActivation = "FastSigmoid";
            public const string ReLUActivation = "ReLU";
            public const string SigmoidActivation = "Sigmoid";
        }

        internal static float GetSuitableFactorForFunction(string function, float countConnections) {
            switch (function) {
                case "FastSigmoid":
                    return 1;
                case "ReLU":
                    return 1;
                case "Sigmoid":
                    return 1 / MathF.Sqrt(countConnections);
            }
            return 1;
        }
    }
}
