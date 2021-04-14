using ILGPU;
using ILGPU.Algorithms;

namespace DeepDave.Layer.Kernels {
    internal class SumCalculate {
        internal static void ConvolutionalLayer2D(Index2 currentInput, ArrayView3D<float> weight, ArrayView2D<float> outputPreviousLayerActivated, ArrayView2D<float> sumInput, ArrayView2D<float> bias, ArrayView<float> variables) {
            int radius = (int) variables[new Index1(1)];
            int diameter = radius * 2 + 1;
            var baseIndex = new Index2(currentInput.X - radius, currentInput.Y - radius);
            var xBounds = outputPreviousLayerActivated.Extent.X;
            var yBounds = outputPreviousLayerActivated.Extent.Y;
            float sum = bias[currentInput];
            for (int i = 0; i < diameter; i++) {
                for (int j = 0; j < diameter; j++) {
                    var asosInput = baseIndex.Add(new Index2(i, j));
                    if (asosInput.X < xBounds & asosInput.X >= 0 & asosInput.Y >= 0 & asosInput.Y < yBounds) {
                        sum += weight[new Index3(currentInput, i * diameter + j)] * outputPreviousLayerActivated[asosInput];
                        var w = weight[new Index3(currentInput, i * diameter + j)]; var oA = outputPreviousLayerActivated[asosInput];
                        if (float.IsNaN(w*oA) | float.IsInfinity(w*oA))
                            ;
                    }
                }
            }
            sumInput[currentInput] = sum;
            if (float.IsNaN(sum) | float.IsInfinity(sum))
                ;
        }
        internal static void FullyConnectedLayer2D(Index2 currentInput, ArrayView3D<float> weights, ArrayView2D<float> outputsPreviousLayerActivated, ArrayView2D<float> sumInput, ArrayView2D<float> bias, ArrayView<float> variables) {
            float sum = bias[currentInput];
            for (int x = 0; x < outputsPreviousLayerActivated.Width; x++) {
                for (int y = 0; y < outputsPreviousLayerActivated.Height; y++) {
                    sum += weights[new Index3(currentInput, x * y + x)] * outputsPreviousLayerActivated[new Index2(x, y)];
                }
            }
            sumInput[currentInput] = sum;
        }
    }
}
