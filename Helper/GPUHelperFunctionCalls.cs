using ILGPU;
using ILGPU.Runtime;
using System;

namespace DeepDave.Helper {
    internal partial class GPUHelper {
        internal class Call {
            internal static void NormalizationFunction(
                Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> action,
                Index2 extent, MemoryBuffer2D<float> outputPreviousLayerActivated, MemoryBuffer2D<float> outputActivated, MemoryBuffer<float> variable
                ) {
                action(accelerator.DefaultStream, extent, outputPreviousLayerActivated, outputActivated, variable);
            }
            internal static void ActivationFunction(
                Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> action,
                Index2 extent, MemoryBuffer2D<float> sumInput, MemoryBuffer2D<float> activated, MemoryBuffer<float> variable
                ) {
                action(accelerator.DefaultStream, extent, sumInput, activated, variable);
            }

            internal static void DerivativeFunction(
                Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> action,
                Index2 extent, MemoryBuffer2D<float> sumInput, MemoryBuffer2D<float> derived, MemoryBuffer<float> variable
                ) {
                action(accelerator.DefaultStream, extent, sumInput, derived, variable);
            }

            internal static void SumCalculate(
                Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> action,
                Index2 extent, MemoryBuffer3D<float> weights, MemoryBuffer2D<float> outputPreviousLayerActivated, MemoryBuffer2D<float> sumInput, MemoryBuffer2D<float> bias, MemoryBuffer<float> variables
                ) {
                action(accelerator.DefaultStream, extent, weights, outputPreviousLayerActivated, sumInput, bias, variables);
            }

            internal static void SumError(
                Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView3D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> action,
                Index2 extent, MemoryBuffer2D<float> error, MemoryBuffer2D<float> errorNextLayer, MemoryBuffer3D<float> weightNextLayer, MemoryBuffer2D<float> derived, MemoryBuffer<float> variable
                ) {
                action(accelerator.DefaultStream, extent, error, errorNextLayer, weightNextLayer, derived, dummyBuffer2D, variable);
            }

            internal static void SumError(
                Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView3D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> action,
                Index2 extent, MemoryBuffer2D<float> error, MemoryBuffer2D<float> activated, MemoryBuffer2D<float> should, MemoryBuffer2D<float> derived, MemoryBuffer<float> variable
                ) {
                action(accelerator.DefaultStream, extent, error, activated, dummyBuffer3D, derived, should, variable);
            }

            internal static void WheightAdjustment(
                Action<AcceleratorStream, Index2, ArrayView3D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> action,
                Index2 extent, MemoryBuffer3D<float> weights, MemoryBuffer2D<float> error, MemoryBuffer2D<float> activatedPreviousLayer, MemoryBuffer2D<float> bias, MemoryBuffer<float> variables
                ) {
                action(accelerator.DefaultStream, extent, weights, error, activatedPreviousLayer, bias, variables);
            }
            internal static void Wait() {
                //Thread.Sleep(1);
                accelerator.Synchronize();
            }
        }
    }
}
