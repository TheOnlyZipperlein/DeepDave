using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepDave.Layer.Kernels {
    internal class NormalizationFunctions {
        internal static void ByteToByteFraction(Index2 currentInput, ArrayView2D<float> outputPreviousLayerActivated, ArrayView2D<float> outputActivated, ArrayView<float> variable) {
            outputActivated[currentInput] = outputPreviousLayerActivated[currentInput] / 255;
        }

        internal static void ByteFractionToByte(Index2 currentInput, ArrayView2D<float> outputPreviousLayerActivated, ArrayView2D<float> outputActivated, ArrayView<float> variable) {
            outputActivated[currentInput] = outputPreviousLayerActivated[currentInput] * 255;
        }
    }
}
