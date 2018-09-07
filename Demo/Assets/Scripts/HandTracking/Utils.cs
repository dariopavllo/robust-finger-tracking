using System.Collections.Generic;
using UnityEngine;
using AMath = Accord.Math;

namespace HandTracking
{
    /// <summary>
    /// This class contains some mathematical utilities.
    /// </summary>
    public static class Utils
    {
        /// <summary>
        /// Swap two values.
        /// </summary>
        /// <param name="lhs">Value 1</param>
        /// <param name="rhs">Value 2</param>
        public static void Swap<T>(ref T lhs, ref T rhs)
        {
            T temp;
            temp = lhs;
            lhs = rhs;
            rhs = temp;
        }

        /// <summary>
        /// Compute rigid motion using the singular value decomposition (SVD).
        /// Returns the best transformation that moves X towards Y minimizing the mean squared error (MSE),
        /// such that y = x*R + offset
        /// </summary>
        /// <param name="x">The list of source points (read-only)</param>
        /// <param name="y">The list of destination points (read-only)</param>
        /// <param name="R">The output rotation matrix (3x3)</param>
        /// <param name="offset">The output offset (3D vector)</param>
        public static void ComputeRigidMotion(List<Vector3> x, List<Vector3> y, out double[,] R, out double[] offset)
        {
            if (x.Count == 0)
            {
                // Extreme corner case
                R = AMath.Matrix.Identity(3);
                offset = new double[] { 0, 0, 0 };
                return;
            }

            double[,] X = new double[x.Count, 3];
            double[,] Y = new double[y.Count, 3];
            for (int i = 0; i < x.Count; i++)
            {
                X[i, 0] = x[i][0];
                X[i, 1] = x[i][1];
                X[i, 2] = x[i][2];

                Y[i, 0] = y[i][0];
                Y[i, 1] = y[i][1];
                Y[i, 2] = y[i][2];
            }
            var Xc = Accord.Statistics.Measures.Mean(X, 0);
            var Yc = Accord.Statistics.Measures.Mean(Y, 0);
            var H = AMath.Matrix.TransposeAndDot(AMath.Elementwise.Subtract(Y, Yc, 0), AMath.Elementwise.Subtract(X, Xc, 0));
            var svd = new AMath.Decompositions.SingularValueDecomposition(H);
            R = AMath.Matrix.DotWithTransposed(svd.RightSingularVectors, svd.LeftSingularVectors);
            if (AMath.Matrix.Determinant(R) < 0) // Check whether the determinant is -1
            {
                // We have an improper rotation, i.e. a reflection transformation.
                double[,] V = svd.RightSingularVectors;

                V[0, 2] *= -1;
                V[1, 2] *= -1;
                V[2, 2] *= -1;
                R = AMath.Matrix.DotWithTransposed(V, svd.LeftSingularVectors); // Now the determinant is +1
            }

            offset = AMath.Elementwise.Subtract(Yc, AMath.Matrix.Dot(Xc, R));
        }

        /// <summary>
        /// Compute the transformation y = v*R + offset
        /// </summary>
        /// <param name="v">The point to transform</param>
        /// <param name="R">3x3 rotation matrix</param>
        /// <param name="offset">3D offset</param>
        /// <returns></returns>
        public static Vector3 RotateAndShift(Vector3 v, double[,] R, double[] offset)
        {
            double[] x = new double[] { v.x, v.y, v.z };
            x = AMath.Matrix.Dot(x, R);
            x = AMath.Elementwise.Add(x, offset);
            Vector3 pos;
            pos.x = (float)x[0];
            pos.y = (float)x[1];
            pos.z = (float)x[2];
            return pos;
        }

        /// <summary>
        /// Compute the transformation y = v*R + offset on the given Transform object.
        /// </summary>
        /// <param name="t">The transform to modify</param>
        /// <param name="R">3x3 rotation matrix</param>
        /// <param name="offset">3D offset</param>
        public static void RotateAndShift(Transform t, double[,] R, double[] offset)
        {
            Vector3 pos = t.position;
            Vector3 forward = t.rotation * Vector3.forward + pos;
            Vector3 up = t.rotation * Vector3.up + pos;

            pos = RotateAndShift(pos, R, offset);
            forward = RotateAndShift(forward, R, offset);
            up = RotateAndShift(up, R, offset);

            t.position = pos;
            t.rotation = Quaternion.LookRotation(forward - pos, up - pos);
        }

        public static void AlignDirections(Transform origin, Vector3 sourcePoint, Vector3 destinationPoint)
        {
            var rotation = Quaternion.FromToRotation(sourcePoint - origin.position, destinationPoint - origin.position);
            origin.rotation = rotation * origin.rotation;
        }

        /// <summary>
        /// Compute the inverse of RotateAndShift.
        /// </summary>
        /// <param name="v">The vector to transform.</param>
        /// <param name="R">3x3 rotation matrix</param>
        /// <param name="offset">3D offset</param>
        /// <returns></returns>
        public static Vector3 InverseRotateAndShift(Vector3 v, double[,] R, double[] offset)
        {
            double[] x = new double[] { v.x, v.y, v.z };
            x = AMath.Elementwise.Subtract(x, offset);
            x = AMath.Matrix.DotWithTransposed(x, R); // The inverse of a rotation matrix is its transpose
            Vector3 pos;
            pos.x = (float)x[0];
            pos.y = (float)x[1];
            pos.z = (float)x[2];
            return pos;
        }

        /// <summary>
        /// Compute the weights W to express Y as an affine combination of X,
        /// such that Y = W*X. Each row of W sums up to 1.
        /// </summary>
        /// <param name="x">List of N source points</param>
        /// <param name="y">List of M target points</param>
        /// <returns>An M x N weight matrix</returns>
        public static double[,] ComputeImputationWeights(List<Vector3> x, List<Vector3> y)
        {
            double[,] X = new double[x.Count, 4];
            for (int i = 0; i < x.Count; i++)
            {
                X[i, 0] = x[i][0];
                X[i, 1] = x[i][1];
                X[i, 2] = x[i][2];
                X[i, 3] = 1;
            }

            double[,] Y = new double[y.Count, 4];
            for (int i = 0; i < y.Count; i++)
            {
                Y[i, 0] = y[i][0];
                Y[i, 1] = y[i][1];
                Y[i, 2] = y[i][2];
                Y[i, 3] = 1;
            }

            // L2 regularization constant
            const double lambda = 1e-8;

            // W = Y * X^T * (X * X^T + lambda*I)^(-1)
            double[,] W = AMath.Matrix.Transpose(AMath.Matrix.Solve(
                    AMath.Elementwise.Add(AMath.Matrix.DotWithTransposed(X, X), AMath.Matrix.Diagonal(x.Count, lambda)),
                    AMath.Matrix.DotWithTransposed(X, Y)
                ));

            return W;
        }

        /// <summary>
        /// Compute Y = W*X, given W and X.
        /// </summary>
        /// <param name="x">The known N points</param>
        /// <param name="weights">The weight matrix of size M x N</param>
        /// <returns>A list of M predicted points</returns>
        public static Vector3[] Impute(List<Vector3> x, double[,] weights)
        {
            double[,] X = new double[x.Count, 3];
            for (int i = 0; i < x.Count; i++)
            {
                X[i, 0] = x[i][0];
                X[i, 1] = x[i][1];
                X[i, 2] = x[i][2];
            }

            double[,] Y = AMath.Matrix.Dot(weights, X);
            Vector3[] output = new Vector3[weights.GetLength(0)];
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                output[i].x = (float)Y[i, 0];
                output[i].y = (float)Y[i, 1];
                output[i].z = (float)Y[i, 2];
            }

            return output;
        }
    }

}