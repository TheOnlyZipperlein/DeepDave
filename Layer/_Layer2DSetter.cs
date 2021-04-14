using System;
using System.Collections.Generic;
using System.Text;

namespace DeepDave.Layer {
    public abstract partial class Layer2D { 
        protected virtual void SetPrevious(Layer2D prevLayer) {
            this.prevLayer = prevLayer;
        }
        protected virtual void SetNext(Layer2D nextLayer) {
            this.nextLayer = nextLayer;
        }
    }
}
