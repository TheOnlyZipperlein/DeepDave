using ILGPU;
using ILGPU.Algorithms;

namespace DeepDave.Layer.Kernels {
    internal class DerivativeFunctions {
        internal static void FastSigmoid(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> derived) {
            float v = sumInput[currentInput]/10;
            if (v < 0) v *= -1;
            v = 1 / (2 * (v + 1) * (v + 1));
            derived[currentInput] = v;
        }

        internal static void ReLU(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> derived) {
            float v;
            if (sumInput[currentInput] < 0) v = 0.01f;            
            else v = 1.0f;
            derived[currentInput] = v;
        }

        internal static void Sigmoid(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> derived) {
            float v = System.MathF.Exp(-sumInput[currentInput]/ 2560);
            var d = v / (2560 * v * v + 2 * 2560 + v + 2560);
            derived[currentInput] = d;
            if (derived[currentInput] > 10)
                ;
            if (derived[currentInput] < -10)
                ;
        }        
    }
}
