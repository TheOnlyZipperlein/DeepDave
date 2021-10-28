using ILGPU;

namespace DeepDave.Layer.Kernels {
    internal class SumErrorOut {
        internal static void Common(Index2 currentInput, ArrayView2D<float> error, ArrayView2D<float> activated, ArrayView3D<float> weightNextLayer, ArrayView2D<float> derived, ArrayView2D<float> should, ArrayView<float> variable) {
            error[currentInput] = (activated[currentInput] - should[currentInput]) * derived[currentInput];
            var e = error[currentInput]; var a = activated[currentInput]; var s = should[currentInput]; var d = derived[currentInput];
            var c = 0;
            while (error[currentInput] > 1 | error[currentInput] < -1) {
                error[currentInput] /= 10;
                c++;
                if (c > 20) {
                    ;
                }
            }
            if (float.IsNaN(e) | float.IsInfinity(e) | e < -100 | e > 100)
                ;
        }
    }
}
