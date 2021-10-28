using ILGPU;
using System;

namespace DeepDave.Layer.Kernels {
    internal class ActivationFunctions {
        internal static void FastSigmoid(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> activated, ArrayView<float> variable) {
            float sumAbs = sumInput[currentInput] / 10;
            if (sumAbs < 0) sumAbs *= -1;
            activated[currentInput] = 0.5f * (sumInput[currentInput] / 10 / (1 + sumAbs)) + 0.5f;
        }
        internal static void ReLU(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> activated, ArrayView<float> variable) {
            if (sumInput[currentInput] < 0) {
                activated[currentInput] = sumInput[currentInput] * 0.01f;
            } else {
                activated[currentInput] = sumInput[currentInput];
            }
        }
        internal static void Sigmoid(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> activated, ArrayView<float> variable) {
            activated[currentInput] = 1 / (1 + MathF.Exp((-sumInput[currentInput] / 2560)));
        }
        internal static void Softmax(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> activated, ArrayView<float> variable) {
            activated[currentInput] = MathF.Exp(sumInput[currentInput]);
        }

    }
}
