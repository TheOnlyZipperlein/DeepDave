using ILGPU;
using ILGPU.Algorithms;

namespace DeepDave.Layer.Kernels {
    internal class WheightAdjustment {
        internal static void ConvolutionalLayer2D(Index2 currentInput, ArrayView3D<float> weights, ArrayView2D<float> error, ArrayView2D<float> activatedPreviousLayer, ArrayView2D<float> bias, ArrayView<float> variables) {
            int radius = (int) variables[new Index1(1)];
            int diameter = radius * 2 + 1;
            var fac = error[currentInput] * variables[new Index1(0)];
            float xBounds = error.Extent.X;
            float yBounds = error.Extent.Y;
            bias[currentInput] -= fac;
            var baseIndex = new Index2(currentInput.X - radius, currentInput.Y - radius);
            for (int i = 0; i < diameter; i++) {
                for (int j = 0; j < diameter; j++) {
                    var asosInput = baseIndex.Add(new Index2(i, j));
                    if (asosInput.X < xBounds & asosInput.X >= 0 & asosInput.Y >= 0 & asosInput.Y < yBounds) {
                        var adjustment = fac * activatedPreviousLayer[asosInput];
                        weights[new Index3(currentInput, i * diameter + j)] -= adjustment;                        
                    }
                }
            }
        }
        internal static void FullyConnectedLayer2D(Index2 currentInput, ArrayView3D<float> weights, ArrayView2D<float> error, ArrayView2D<float> activatedPreviousLayer, ArrayView2D<float> bias, ArrayView<float> variables) {
            var fac = error[currentInput] * variables[new Index1(0)];
            bias[currentInput] -= fac;
            for (int x = 0; x < activatedPreviousLayer.Width; x++) {
                for (int y = 0; y < activatedPreviousLayer.Height; y++) {
                    Index2 asosIndex = new Index2(x, y);
                    var adjustment = fac * activatedPreviousLayer[asosIndex];
                    if (activatedPreviousLayer[asosIndex] != 0)
                        ;
                    weights[new Index3(currentInput, x * y + x)] -= adjustment;
                }
            }
        }

    }
}
