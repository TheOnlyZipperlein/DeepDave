using DeepDave.Helper;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace DeepDave.Layer {
    public abstract partial class Layer2D {
        internal virtual void CalculateOutput() {
            var accelerator = GPUHelper.accelerator;
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                GPUHelper.Call.SumCalculate(sumCalculate, this.GetUnactivatedBuffer(currentSlice).Extent, this.GetWeightBuffer(currentSlice), prevLayer.GetActivatedBuffer(currentSlice), this.GetUnactivatedBuffer(currentSlice), bias[currentSlice], variables);
            }
            GPUHelper.Call.Wait();
            this.ActivateOutput();
        }
        internal virtual void ActivateOutput() {
            var accelerator = GPUHelper.accelerator;
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                GPUHelper.Call.DerivativeFunction(derivation, this.GetDerived(currentSlice).Extent, this.GetUnactivatedBuffer(currentSlice), this.GetDerived(currentSlice));
                GPUHelper.Call.ActivationFunction(activation, this.GetActivatedBuffer(currentSlice).Extent, this.GetUnactivatedBuffer(currentSlice), this.GetActivatedBuffer(currentSlice));
            }
            GPUHelper.Call.Wait();
        }

        internal virtual void CalculateError(MemoryBuffer2D<float>[] shoulds) {
            var accelerator = GPUHelper.accelerator;
            if (sumErrorFunction == null) return;
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                if (nextLayer == null)
                    GPUHelper.Call.SumError(sumForError, this.GetErrors(currentSlice).Extent, this.GetErrors(currentSlice), this.GetActivatedBuffer(currentSlice), shoulds[currentSlice], this.GetDerived(currentSlice), variables);
                else
                    GPUHelper.Call.SumError(sumForError, this.GetErrors(currentSlice).Extent, this.GetErrors(currentSlice), nextLayer.GetErrors(currentSlice), nextLayer.GetWeightBuffer(currentSlice), this.GetDerived(currentSlice), variables);
            }
            GPUHelper.Call.Wait();
            prevLayer.CalculateError(null);
        }
        internal virtual void AdjustWeights() {
            var accelerator = GPUHelper.accelerator;
            if (adjustWheigts == null) return;
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                GPUHelper.Call.WheightAdjustment(adjustWheigts, this.GetActivatedBuffer(currentSlice).Extent, this.GetWeightBuffer(currentSlice), this.GetErrors(currentSlice), prevLayer.GetActivatedBuffer(currentSlice), this.bias[currentSlice], this.variables);
            }
            GPUHelper.Call.Wait();
            prevLayer.AdjustWeights();
        }
    }
}
