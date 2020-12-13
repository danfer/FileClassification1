using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace FileClassification.ML.Objects
{
    public class FilePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsMalicious { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
