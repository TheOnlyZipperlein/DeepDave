using DeepDave.Helper;
using ILGPU.Runtime;

namespace DeepDave.Layer {
    public abstract partial class Layer2D {
        internal void CalculateOutput(MemoryBuffer2D<float>[] shoulds = null) {
            if (prevLayer != null) prevLayer.CalculateOutput();
            this.PreCalculate();
            this.CalculateOutput_();
            this.PostCalculate();
            this.PreActivation();
            this.ActivateOutput();
            this.PostActivation();
            this.PreDerivation();
            this.DerivateOutput();
            this.PostDerivation();
            if (shoulds != null) {
                this.CalculateError(shoulds);
                this.AdjustWeights();
            }
        }
        /// <summary>
        /// Calls the loaded "sumCalculate" first.
        /// </summary>
        internal virtual void CalculateOutput_() {
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                GPUHelper.Call.SumCalculate(sumCalculate, this.GetUnactivatedBuffer(currentSlice).Extent, this.GetWeightBuffer(currentSlice), prevLayer.GetActivatedBuffer(currentSlice), this.GetUnactivatedBuffer(currentSlice), bias[currentSlice], variable[currentSlice]);
            }
            GPUHelper.Call.Wait();
        }
        internal virtual void PreCalculate() { }
        internal virtual void PostCalculate() { }
        /// <summary>
        /// Calls the loaded "activation" Method.
        /// </summary>
        internal virtual void ActivateOutput() {
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                GPUHelper.Call.ActivationFunction(activation, this.GetActivatedBuffer(currentSlice).Extent, this.GetUnactivatedBuffer(currentSlice), this.GetActivatedBuffer(currentSlice), this.variable[currentSlice]);
            }
            GPUHelper.Call.Wait();
        }
        internal virtual void PreActivation() { }
        internal virtual void PostActivation() { }
        /// <summary>
        /// Calls the loaded "derivation" Method.
        /// </summary>
        internal virtual void DerivateOutput() {
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                GPUHelper.Call.DerivativeFunction(derivation, this.GetDerived(currentSlice).Extent, this.GetUnactivatedBuffer(currentSlice), this.GetDerived(currentSlice), this.variable[currentSlice]);
            }
            GPUHelper.Call.Wait();
        }
        internal virtual void PreDerivation() { }
        internal virtual void PostDerivation() { }
        /******************************************************************************************************************************************************************/
        internal void CalculateError(MemoryBuffer2D<float>[] shoulds) {
            this.PreError(shoulds);
            this.CalculateError_(shoulds);
            this.PostError(shoulds);
            if (prevLayer != null) prevLayer.CalculateError(null);
        }
        /// <summary>
        /// Calls the loaded "sumForError" Method.
        /// </summary>
        /// <param name="shoulds">How the output should look like for the given input.</param>
        internal virtual void CalculateError_(MemoryBuffer2D<float>[] shoulds) {
            if (sumErrorFunction == null) return;
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                if (nextLayer == null)
                    GPUHelper.Call.SumError(sumForError, this.GetErrors(currentSlice).Extent, this.GetErrors(currentSlice), this.GetActivatedBuffer(currentSlice), shoulds[currentSlice], this.GetDerived(currentSlice), variable[currentSlice]);
                else
                    GPUHelper.Call.SumError(sumForError, this.GetErrors(currentSlice).Extent, this.GetErrors(currentSlice), nextLayer.GetErrors(currentSlice), nextLayer.GetWeightBuffer(currentSlice), this.GetDerived(currentSlice), variable[currentSlice]);
            }
            GPUHelper.Call.Wait();
        }
        internal virtual void PreError(MemoryBuffer2D<float>[] shoulds) { }
        internal virtual void PostError(MemoryBuffer2D<float>[] shoulds) { }
        /******************************************************************************************************************************************************************/
        internal void AdjustWeights() {
            PreAdjust();
            AdjustWeights_();
            PostAdjust();
            if (prevLayer != null) prevLayer.AdjustWeights();
        }
        /// <summary>
        /// Calls the loaded "adjustWheigts" Method.
        /// </summary>
        internal virtual void AdjustWeights_() {
            if (adjustWheigts == null) return;
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                GPUHelper.Call.WheightAdjustment(adjustWheigts, this.GetActivatedBuffer(currentSlice).Extent, this.GetWeightBuffer(currentSlice), this.GetErrors(currentSlice), prevLayer.GetActivatedBuffer(currentSlice), this.bias[currentSlice], this.variable[currentSlice]);
            }
            GPUHelper.Call.Wait();
        }
        internal virtual void PreAdjust() { }
        internal virtual void PostAdjust() { }
    }
}
