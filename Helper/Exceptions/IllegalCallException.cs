using System;

namespace DeepDave.Helper.Exceptions {
    public class IllegalCallException : Exception {
        public IllegalCallException(String error) : base(error) {
        }
    }
}
