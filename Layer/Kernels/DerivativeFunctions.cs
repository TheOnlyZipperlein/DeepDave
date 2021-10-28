using ILGPU;
using System;

namespace DeepDave.Layer.Kernels {
    internal class DerivativeFunctions {
        internal static void FastSigmoid(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> derived, ArrayView<float> variable) {
            float v = sumInput[currentInput] / variable[new Index1(1)]; ;
            if (v < 0) v *= -1;
            v = 1 / (2 * (v + 1) * (v + 1));
            derived[currentInput] = v;
        }

        internal static void ReLU(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> derived, ArrayView<float> variable) {
            var fac = variable[new Index1(1)];
            if (sumInput[currentInput] < 0) derived[currentInput] = fac * 0.01f;
            else derived[currentInput] = fac * 1.0f;
        }

        internal static void Sigmoid(Index2 currentInput, ArrayView2D<float> sumInput, ArrayView2D<float> derived, ArrayView<float> variable) {
            var fac = variable[new Index1(1)];
            float v = MathF.Exp(-sumInput[currentInput] / fac);
            var d = v / (fac * v * v + 2 * fac + v + fac);
            derived[currentInput] = d;
        }


        internal static void Softmax(Index2 currentInput, ArrayView2D<float> activated, ArrayView2D<float> derived) {
            derived[currentInput] = (1 - activated[currentInput]) * activated[currentInput];
        }
    }
}
