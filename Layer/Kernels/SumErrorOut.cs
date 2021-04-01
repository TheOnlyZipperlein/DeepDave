using ILGPU;
using ILGPU.Algorithms;

namespace DeepDave.Layer.Kernels {
    internal class SumErrorOut {
        internal static void Common(Index2 currentInput, ArrayView2D<float> error, ArrayView2D<float> activated, ArrayView3D<float> weightNextLayer, ArrayView2D<float> derived, ArrayView2D<float> should, ArrayView<float> variable) {
            error[currentInput] = (activated[currentInput] - should[currentInput]) * derived[currentInput];
        }
    }
}
