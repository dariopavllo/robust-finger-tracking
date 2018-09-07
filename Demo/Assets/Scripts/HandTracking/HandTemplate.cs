using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace HandTracking
{
    /// <summary>
    /// Represents a hand template.
    /// This class is used to keep the correspondence between markers and joints.
    /// </summary>
    public class HandTemplate
    {
        private List<string> markerList;
        private List<string> alignmentMarkerList;
        private Dictionary<string, string> jointMap;
        private Dictionary<string, Vector3> offsetMap;
        private Dictionary<string, Vector3> bonePositionMap;
        private Dictionary<string, Vector3> markerPositionMap;
        private Dictionary<string, string> markerMap;
        private string pivot;

        /// <summary>
        /// Get the ordered list of marker IDs in this template.
        /// </summary>
        public IList<string> MarkerList
        {
            get
            {
                return markerList.AsReadOnly();
            }
        }

        /// <summary>
        /// Get the ordered list of alignment marker IDs in this template.
        /// </summary>
        public IList<string> AlignmentMarkerList
        {
            get
            {
                return alignmentMarkerList.AsReadOnly();
            }
        }

        public HandTemplate()
        {
            markerList = new List<string>();
            alignmentMarkerList = new List<string>();
            jointMap = new Dictionary<string, string>();
            bonePositionMap = new Dictionary<string, Vector3>();
            offsetMap = new Dictionary<string, Vector3>();
            markerPositionMap = new Dictionary<string, Vector3>();
            markerMap = new Dictionary<string, string>();
        }

        /// <summary>
        /// Add a marker to this hand template. The insertion order is important.
        /// </summary>
        /// <param name="markerId">The ID (alias) of the marker</param>
        /// <param name="jointName">The joint associated with this marker</param>
        /// <param name="offset">Offset in object space, relative to the marker</param>
        /// <param name="useForAlignment">Use this marker for alignment (i.e. pose estimation)</param>
        public void AddMarker(string markerId, string jointName, Vector3 offset, bool useForAlignment = false)
        {
            markerList.Add(markerId);
            jointMap[markerId] = jointName;
            bonePositionMap[markerId] = GetPositionWithOffset(jointName, Vector3.zero);
            markerPositionMap[markerId] = GetPositionWithOffset(jointName, offset);
            offsetMap[markerId] = offset;
            markerMap[jointName] = markerId;

            if (useForAlignment)
            {
                alignmentMarkerList.Add(markerId);
            }
        }

        /// <summary>
        /// Set the pivot (center of rotation) for this hand template.
        /// </summary>
        /// <param name="jointName">Joint corresponding to the pivot</param>
        public void SetPivot(string jointName)
        {
            pivot = jointName;
        }

        public string GetJointName(string markerId)
        {
            return jointMap[markerId];
        }

        public string GetMarkerId(string jointName)
        {
            return markerMap[jointName];
        }

        public Vector3 GetMarkerOffset(string markerId)
        {
            return offsetMap[markerId];
        }

        public Vector3 GetJointOffset(string jointName)
        {
            return offsetMap[markerMap[jointName]];
        }

        public Vector3 GetInitialMarkerWorldPosition(string markerId)
        {
            return markerPositionMap[markerId];
        }

        public Vector3 GetInitialJointWorldPosition(string jointName)
        {
            return bonePositionMap[markerMap[jointName]];
        }

        /// <summary>
        /// Save this template to file, so that it can be loaded by another script
        /// (e.g the training script in Python).
        /// </summary>
        /// <param name="path"></param>
        public void SaveToFile(string path)
        {
            using (var file = new StreamWriter(path))
            {
                // Write header
                file.WriteLine("marker,marker.pos.x,marker.pos.y,marker.pos.z,joint,joint.pos.x,joint.pos.y,joint.pos.z,alignment");

                Vector3 pivotPos = GetInitialJointWorldPosition(pivot);
                foreach (var marker in markerList)
                {
                    // Write joint positions
                    string jointName = GetJointName(marker);
                    Vector3 markerPos = GetInitialMarkerWorldPosition(marker) - pivotPos;
                    Vector3 jointPos = GetInitialJointWorldPosition(jointName) - pivotPos;
                    string[] strings = new string[]
                    {
                        marker.ToString(),
                        markerPos.x.ToString(CultureInfo.InvariantCulture),
                        markerPos.y.ToString(CultureInfo.InvariantCulture),
                        markerPos.z.ToString(CultureInfo.InvariantCulture),
                        jointName,
                        jointPos.x.ToString(CultureInfo.InvariantCulture),
                        jointPos.y.ToString(CultureInfo.InvariantCulture),
                        jointPos.z.ToString(CultureInfo.InvariantCulture),
                        alignmentMarkerList.Contains(marker) ? "1" : "0"
                    };

                    file.WriteLine(string.Join(",", strings));
                }
            }
        }

        /// <summary>
        /// Compute a rigid motion on the given points, using the information memorized in this hand template.
        /// This method computes the transformation: transformPoints = (transformPoints * R) + offset
        /// </summary>
        /// <param name="alignmentPoints">The marker positions that are used for alignment.</param>
        /// <param name="transformPoints">The marker positions that need to be moved (they will be transformed in place).</param>
        /// <param name="R">Output rotation matrix (3x3)</param>
        /// <param name="offset">Output offset (3D)</param>
        public void AlignTransform(Dictionary<string, Vector3> alignmentPoints, Dictionary<string, Vector3> transformPoints, out double[,] R, out double[] offset)
        {
            var X = new List<Vector3>();
            var Y = new List<Vector3>();
            Vector3 pivotPos = GetInitialJointWorldPosition(pivot);
            foreach (var point in alignmentPoints)
            {
                X.Add(point.Value);
                Y.Add(GetInitialMarkerWorldPosition(point.Key) - pivotPos);
            }
            Utils.ComputeRigidMotion(X, Y, out R, out offset);
            foreach (var point in transformPoints.ToList())
            {
                transformPoints[point.Key] = Utils.RotateAndShift(point.Value, R, offset);
            }
        }

        /// <summary>
        /// Computes an inverse rigid motion transformation.
        /// </summary>
        /// <param name="points">The points that must be moved (they will be transformed in place).</param>
        /// <param name="R">Rotation matrix (3x3)</param>
        /// <param name="offset">Offset (3D)</param>
        public void InverseAlignTransform(Dictionary<string, Vector3> points, double[,] R, double[] offset)
        {
            foreach (var point in points.ToList())
            {
                points[point.Key] = Utils.InverseRotateAndShift(point.Value, R, offset);
            }
        }

        private Vector3 GetPositionWithOffset(string jointName, Vector3 offset)
        {
            Transform tBone = GameObject.Find(jointName).transform;
            return tBone.TransformPoint(offset);
        }
    }

}