/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package edu.gatech.cse8803.clustering

import breeze.linalg.{DenseVector => DBV, DenseMatrix => DBM, diag, max, eigSym}
import org.apache.spark.mllib.linalg.{Matrices, Vectors, Matrix, Vector}
import org.apache.spark.rdd.RDD

/**
 * @param weight Weights for each Gaussian distribution in the mixture, where weight(i) is
 *               the weight for Gaussian i, and weight.sum == 1
 * @param mu Means for each Gaussian in the mixture, where mu(i) is the mean for Gaussian i
 * @param sigma Covariance maxtrix for each Gaussian in the mixture, where sigma(i) is the
 *              covariance matrix for Gaussian i
 */
class GaussianMixtureModel(
                            val weights: Array[Double],
                            val gaussians: Array[MultivariateGaussian]) extends Serializable {

  require(weights.length == gaussians.length, "Length of weight and Gaussian arrays must match")

  /** Number of gaussians in mixture */
  def k: Int = weights.length

  /** Maps given point to their cluster indices. */
  def predict(point: Vector): Int = {
    val r = this.predictSoft(point)
    r.indexOf(r.max)
  }

  /** Maps given points to their cluster indices. */
  def predict(points: RDD[Vector]): RDD[Int] = {
    val responsibilityMatrix = predictSoft(points)
    responsibilityMatrix.map(r => r.indexOf(r.max))
  }

  /**
   * Given the input vector, return the membership value of  vector
   * to all mixture components.
   */
  def predictSoft(point: Vector): Array[Double] = {
    computeSoftAssignments(toBreezeVector(point).toDenseVector, gaussians, weights, k)
  }

  /**
   * Given the input vectors, return the membership value of each vector
   * to all mixture components.
   */
  def predictSoft(points: RDD[Vector]): RDD[Array[Double]] = {
    val sc = points.sparkContext
    val bcDists = sc.broadcast(gaussians)
    val bcWeights = sc.broadcast(weights)
    points.map { x =>
      computeSoftAssignments(toBreezeVector(x).toDenseVector, bcDists.value, bcWeights.value, k)
    }
  }

  /**
   * Compute the partial assignments for each vector
   */
  private def computeSoftAssignments(
                                      pt: DBV[Double],
                                      dists: Array[MultivariateGaussian],
                                      weights: Array[Double],
                                      k: Int): Array[Double] = {
    val p = weights.zip(dists).map {
      case (weight, dist) => 2E16 + weight * dist.pdf(pt)
    }
    val pSum = p.sum
    for (i <- 0 until k) {
      p(i) /= pSum
    }
    p
  }
}

class MultivariateGaussian (
                             val mu: Vector,
                             val sigma: Matrix) extends Serializable {

  require(sigma.numCols == sigma.numRows, "Covariance matrix must be square")
  require(mu.size == sigma.numCols, "Mean vector length must match covariance matrix size")

  private val breezeMu = toBreezeVector(mu).toDenseVector

  def this(mu: DBV[Double], sigma: DBM[Double]) = {
    this(fromBreeze(mu), fromBreeze(sigma))
  }
  /**
   * Compute distribution dependent constants:
   *    rootSigmaInv = D^(-1/2)^ * U, where sigma = U * D * U.t
   *    u = log((2*pi)^(-k/2)^ * det(sigma)^(-1/2)^)
   */
  private val (rootSigmaInv: DBM[Double], u: Double) = calculateCovarianceConstants

  /** Returns density of this multivariate Gaussian at given point, x */
  def pdf(x: Vector): Double = {
    pdf(toBreezeVector(x).toDenseVector)
  }

  /** Returns the log-density of this multivariate Gaussian at given point, x */
  def logpdf(x: Vector): Double = {
    logpdf(toBreezeVector(x).toDenseVector)
  }

  /** Returns density of this multivariate Gaussian at given point, x */
  def pdf(x: DBV[Double]): Double = {
    math.exp(logpdf(x))
  }

  /** Returns the log-density of this multivariate Gaussian at given point, x */
  def logpdf(x: DBV[Double]): Double = {
    val delta = x - breezeMu
    val v = rootSigmaInv * delta
    u + v.t * v * -0.5
  }

  /**
   * Calculate distribution dependent components used for the density function:
   *    pdf(x) = (2*pi)^(-k/2)^ * det(sigma)^(-1/2)^ * exp((-1/2) * (x-mu).t * inv(sigma) * (x-mu))
   * where k is length of the mean vector.
   *
   * We here compute distribution-fixed parts
   *  log((2*pi)^(-k/2)^ * det(sigma)^(-1/2)^)
   * and
   *  D^(-1/2)^ * U, where sigma = U * D * U.t
   *
   * Both the determinant and the inverse can be computed from the singular value decomposition
   * of sigma.  Noting that covariance matrices are always symmetric and positive semi-definite,
   * we can use the eigendecomposition. We also do not compute the inverse directly; noting
   * that
   *
   *    sigma = U * D * U.t
   *    inv(Sigma) = U * inv(D) * U.t
   *               = (D^{-1/2}^ * U).t * (D^{-1/2}^ * U)
   *
   * and thus
   *
   *    -0.5 * (x-mu).t * inv(Sigma) * (x-mu) = -0.5 * norm(D^{-1/2}^ * U  * (x-mu))^2^
   *
   * To guard against singular covariance matrices, this method computes both the
   * pseudo-determinant and the pseudo-inverse (Moore-Penrose).  Singular values are considered
   * to be non-zero only if they exceed a tolerance based on machine precision, matrix size, and
   * relation to the maximum singular value (same tolerance used by, e.g., Octave).
   */
  private def calculateCovarianceConstants: (DBM[Double], Double) = {
    val eigSym.EigSym(d, u) = eigSym(toBreezeMatrix(sigma).toDenseMatrix) // sigma = u * diag(d) * u.t

    // For numerical stability, values are considered to be non-zero only if they exceed tol.
    // This prevents any inverted value from exceeding (eps * n * max(d))^-1
    val tol = 2E16 * max(d) * d.length

    try {
      // log(pseudo-determinant) is sum of the logs of all non-zero singular values
      val logPseudoDetSigma = d.activeValuesIterator.filter(_ > tol).map(math.log).sum

      // calculate the root-pseudo-inverse of the diagonal matrix of singular values
      // by inverting the square root of all non-zero values
      val pinvS = diag(new DBV(d.map(v => if (v > tol) math.sqrt(1.0 / v) else 0.0).toArray))

      (pinvS * u, -0.5 * (mu.size * math.log(2.0 * math.Pi) + logPseudoDetSigma))
    } catch {
      case uex: UnsupportedOperationException =>
        throw new IllegalArgumentException("Covariance matrix has no non-zero singular values")
    }
  }
}
