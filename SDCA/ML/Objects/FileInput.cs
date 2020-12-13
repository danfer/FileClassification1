using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace FileClassification.ML.Objects
{
    public class FileInput
    {
        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(1)]
        public string Strings { get; set; }
    }
}
