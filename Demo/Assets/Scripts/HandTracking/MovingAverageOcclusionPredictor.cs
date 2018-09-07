using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace HandTracking
{
    class MovingAverageOcclusionPredictor : OcclusionManager.Implementation
    {
        private Dictionary<string, Queue<Vector3>> speeds;
        private int windowSize;

        public MovingAverageOcclusionPredictor(int windowSize, HandTemplate handTemplate)
        {
            speeds = new Dictionary<string, Queue<Vector3>>();
            foreach (var marker in handTemplate.MarkerList)
            {
                speeds[marker] = new Queue<Vector3>(windowSize);
            }
            this.windowSize = windowSize;
        }

        public Dictionary<string, Vector3> Predict(Dictionary<string, Vector3> availableMarkers, HashSet<string> occludedMarkers,
            Dictionary<string, Vector3> oldMarkers, HashSet<string> oldAvailableMarkers, float timeSinceLastFrame)
        {
            var output = new Dictionary<string, Vector3>();

            foreach (var marker in availableMarkers)
            {
                var record = speeds[marker.Key];
                record.Enqueue(marker.Value - oldMarkers[marker.Key]); // Delta
                while (record.Count > windowSize)
                {
                    record.Dequeue();
                }
                output.Add(marker.Key, marker.Value);
            }

            foreach (string marker in occludedMarkers)
            {
                Vector3 avgSpeed = Vector3.zero;
                var q = speeds[marker];
                foreach (var elem in q)
                {
                    avgSpeed += elem / q.Count;
                }
                output[marker] = oldMarkers[marker] + avgSpeed;
            }

            return output;
        }
    }

}