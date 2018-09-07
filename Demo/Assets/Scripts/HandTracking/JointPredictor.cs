using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace HandTracking
{
    /// <summary>
    /// This class predicts the angles of joints using a neural network.
    /// </summary>
    public class JointPredictor
    {
        private HandTemplate handTemplate;
        private NeuralNetwork model;

        private List<string> features;
        private List<Vector3> featureComponents;

        /// <summary>
        /// Initialize this joint predictor.
        /// </summary>
        /// <param name="modelFileName">The file name of the neural network model</param>
        /// <param name="handTemplate">The hand template associated with this predictor</param>
        public JointPredictor(string modelFileName, HandTemplate handTemplate)
        {
            model = new NeuralNetwork(modelFileName);
            this.handTemplate = handTemplate;

            features = new List<string>();
            featureComponents = new List<Vector3>();
        }

        /// <summary>
        /// Add a joint to this predictor. The insertion order is important.
        /// </summary>
        /// <param name="jointName">The name of the joint.</param>
        /// <param name="degreesOfFreedom">A 3D vector that describes the degrees of freedom of this joint.
        /// Each component (x, y, z) must be set to either 0 or 1, where 0 indicates that rotation in that axis
        /// is not allowed, whereas 1 indicates that it is. The configuration must be matched to how the neural network
        /// has been trained.</param>
        public void AddJoint(string jointName, Vector3 degreesOfFreedom)
        {
            for (int i = 0; i < 3; i++)
            {
                if (degreesOfFreedom[i] != 0 && degreesOfFreedom[i] != 1)
                {
                    throw new ArgumentException("A degree of freedom must be set to either 0 or 1");
                }
            }
            features.Add(jointName);
            featureComponents.Add(degreesOfFreedom);
        }

        /// <summary>
        /// Predict the angles of all joints.
        /// </summary>
        /// <param name="points">The positions of ALL markers. No missing values are allowed.</param>
        /// <returns>The rotation of each joint.</returns>
        public Dictionary<string, Quaternion> Predict(Dictionary<string, Vector3> points)
        {
            float[] input = ExtractInputFeatures(points);
            float[] output = model.Predict(input);
            return ExtractOutput(output);
        }

        private float[] ExtractInputFeatures(Dictionary<string, Vector3> points)
        {
            points = new Dictionary<string, Vector3>(points); // Defensive copy
            var alignmentPoints = new Dictionary<string, Vector3>();
            foreach (var id in handTemplate.AlignmentMarkerList)
            {
                alignmentPoints.Add(id, points[id]);
                Debug.Assert(!float.IsNaN(points[id].x), "Alignment marker is NaN");
            }
            double[,] R;
            double[] offset;
            handTemplate.AlignTransform(alignmentPoints, points, out R, out offset);

            var markerList = handTemplate.MarkerList;
            float[] features = new float[markerList.Count * 3];
            for (int i = 0; i < markerList.Count; i++)
            {
                Vector3 pos = points[markerList[i]];
                features[3 * i + 0] = pos.x;
                features[3 * i + 1] = pos.y;
                features[3 * i + 2] = pos.z;
                if (float.IsNaN(pos.x))
                {
                    Debug.Assert(!float.IsNaN(pos.x), "NaNs detected in the neural network input (might happen during warmup)");
                    features[3 * i + 0] = features[3 * i + 1] = features[3 * i + 2] = 0;
                }
            }
            return features;
        }

        private Dictionary<string, Quaternion> ExtractOutput(float[] output)
        {
            Debug.Assert(featureComponents.Count == features.Count);

            var result = new Dictionary<string, Quaternion>();
            int outputCount = 0;
            for (int i = 0; i < features.Count; i++)
            {
                Vector3 angles = Vector3.zero;
                for (int j = 0; j < 3; j++)
                {
                    if (featureComponents[i][j] == 1)
                    {
                        angles[j] = output[outputCount] * 180.0f;
                        Debug.Assert(!float.IsNaN(output[outputCount]), "NaNs detected in the neural network output");
                        outputCount++;
                    }
                }
                result[features[i]] = Quaternion.Euler(angles);
            }
            Debug.Assert(outputCount == output.Length, "The feature count is wrong");

            return result;
        }


    }

}