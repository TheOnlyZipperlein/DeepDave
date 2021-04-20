using ILGPU;
using ILGPU.Algorithms;

namespace DeepDave.Kernel {
    internal class SumErrorHidden {
        internal static void ConvolutionalLayer2D(Index2 currentInput, ArrayView2D<float> error, ArrayView2D<float> errorNextLayer, ArrayView3D<float> weightNextLayer, ArrayView2D<float> derived, ArrayView2D<float> should, ArrayView<float> variable) {
            int radius = (int)variable[new Index1(2)];
            int diameter = radius * 2 + 1;
            int xBounds = (int)errorNextLayer.Extent.X;
            int yBounds = (int)errorNextLayer.Extent.Y;
            var baseIndex = new Index2(currentInput.X - radius, currentInput.Y - radius);
            float sum = 0.0f;
            for (int i = 0; i < diameter; i++) {
                for (int j = 0; j < diameter; j++) {
                    var asosInput = baseIndex.Add(new Index2(i, j));
                    if (asosInput.X < xBounds & asosInput.X >= 0 & asosInput.Y >= 0 & asosInput.Y < yBounds) {
                        sum += errorNextLayer[asosInput] * weightNextLayer[new Index3(currentInput, i * diameter + j)];
                    }
                }
            }
            error[currentInput] = sum * derived[currentInput];
            var d = derived[currentInput]; var e = error[currentInput];
            int c = 0;
            while (error[currentInput] > 1 | error[currentInput] < -1) {
                error[currentInput] /= 10;
                c++;
                if (c > 20) {
                    ;
                }
            }
            if (float.IsNaN(error[currentInput]) | float.IsInfinity(e) | sum>1000 | sum <- 1000)
                ;
        }
        internal static void FullyConnectedLayer2D(Index2 currentInput, ArrayView2D<float> error, ArrayView2D<float> errorNextLayer, ArrayView3D<float> weightNextLayer, ArrayView2D<float> derived, ArrayView2D<float> should, ArrayView<float> variable) {
            int xBounds = (int)errorNextLayer.Extent.X;
            int yBounds = (int)errorNextLayer.Extent.Y;
            float sum = 0.0f;
            for (int x = 0; x < xBounds; x++) {
                for (int y = 0; y < yBounds; y++) {
                    Index2 varIndex = new Index2(x, y);
                    sum += errorNextLayer[varIndex] * weightNextLayer[new Index3(varIndex, currentInput.X * currentInput.Y + currentInput.X)];
                }
            }
            error[currentInput] = sum * derived[currentInput];
        }

        internal static void SoftmaxLayer2D(Index2 currentInput, ArrayView2D<float> error, ArrayView2D<float> errorNextLayer, ArrayView3D<float> weightNextLayer, ArrayView2D<float> derived, ArrayView2D<float> should, ArrayView<float> variable) {
            int xBounds = (int)errorNextLayer.Extent.X;
            int yBounds = (int)errorNextLayer.Extent.Y;
            float sum = 0.0f;
            for (int x = 0; x < xBounds; x++) {
                for (int y = 0; y < yBounds; y++) {
                    Index2 varIndex = new Index2(x, y);
                    sum += errorNextLayer[varIndex] * weightNextLayer[new Index3(varIndex, currentInput.X * currentInput.Y + currentInput.X)];
                }
            }
            error[currentInput] = sum * derived[currentInput];
        }
    }
}
