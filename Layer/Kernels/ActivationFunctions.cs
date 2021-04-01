using ILGPU;
using ILGPU.Algorithms;

namespace DeepDave.Layer.Kernels {
    internal class ActivationFunctions {
        internal static void FastSigmoid(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> activated) { 
            float sumAbs = sumInput[currentInput]/10;
            if (sumAbs < 0) sumAbs *= -1;
            activated[currentInput] = 0.5f * (sumInput[currentInput]/10 / (1 + sumAbs)) + 0.5f;
        }        
        internal static void ReLU(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> activated) {
            if (sumInput[currentInput] < 0) {
                activated[currentInput] = sumInput[currentInput]*0.01f;
            } else {
                activated[currentInput] = sumInput[currentInput];
            }
        }      
        internal static void Sigmoid(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> activated) {
            activated[currentInput] = 1 / (1 + System.MathF.Exp((-sumInput[currentInput]/ 2560)));
        }      

    }
}
