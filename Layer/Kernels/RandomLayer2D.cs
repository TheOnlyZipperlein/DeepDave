﻿using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepDave.Layer.Kernels {
    internal class Randomfunctions {
        internal static void Calculate(Index2 currentInput, ArrayView3D<float> weight, ArrayView3D<int> connectionInfo, ArrayView2D<float> outputPreviousLayerActivated, ArrayView2D<float> sumInput, ArrayView2D<float> bias, ArrayView<float> variables) {
            float sum = bias[currentInput];
            for(int i=0; i<connectionInfo.Extent.Z; i+=2) {
                int x = connectionInfo[new Index3(currentInput, i)];
                int y = connectionInfo[new Index3(currentInput, i+1)];
                sum += outputPreviousLayerActivated[new Index2(x, y)]*weight[new Index3(currentInput, i/2)];
            }
            sumInput[currentInput] = sum;
        }

        internal static void AlternateWeights() {

        }

        internal static void AlternateConnections() {

        }
    }
}
