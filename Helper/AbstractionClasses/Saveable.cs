using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DeepDave.Helper.AbstractionClasses {
    internal interface Saveable {
        internal void Save(StreamWriter writer);
    }

    
}
