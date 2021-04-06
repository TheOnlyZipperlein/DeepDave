using ILGPU;
using System;
using System.Collections.Generic;
using System.Text;

namespace DeepDave.Layer.Kernels {
    internal class SoftmaxFunctions {
        internal static void Normalization(Index1 currentInput, ArrayView2D<float> sumInput) {
            float xMax = sumInput[new Index2(0, 0)];
            for (int i = 0; i < sumInput.Extent.X; i++) {
                var index = new Index2(i, 0);
                if (sumInput[index] > xMax) xMax = sumInput[index];
            }
            for (int i = 0; i < sumInput.Extent.X; i++) {
                var index = new Index2(i, 0);
                sumInput[index] -= xMax;
            }
        }
        internal static void SumActivatedOutputs(Index1 currentInput, ArrayView<float> vars, ArrayView2D<float> buffer) {
            float sum = 0f;
            for(int i=0; i<buffer.Extent.X; i++) {
                sum += buffer[new Index2(i, 0)];
            }
            vars[new Index1(1)] = sum;
        }

        internal static void DivisionBySumActivatedOutputs(Index2 currentInput, ArrayView<float> vars, ArrayView2D<float> buffer, ArrayView2D<float> activated) {
            activated[currentInput] = buffer[currentInput] / vars[new Index1(1)];
        }
        internal static void Error(Index2 currentInput, ArrayView2D<float> error, ArrayView2D<float> activated, ArrayView3D<float> weightNextLayer, ArrayView2D<float> derived, ArrayView2D<float> should, ArrayView<float> variable) {
            error[currentInput] = 2 * (activated[currentInput] - should[currentInput]) * derived[currentInput];
        }

    }
}
