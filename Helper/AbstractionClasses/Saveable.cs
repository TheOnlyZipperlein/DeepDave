using System.IO;

namespace DeepDave.Helper.AbstractionClasses {
    internal interface Saveable {
        internal void Save(StreamWriter writer);
    }


}
