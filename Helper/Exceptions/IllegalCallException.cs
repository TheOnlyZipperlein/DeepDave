using System;
using System.Collections.Generic;
using System.Text;

namespace DeepDave.Helper.Exceptions {
    public class IllegalCallException : Exception {
        public IllegalCallException(String error) : base(error) {
        }
    }
}
