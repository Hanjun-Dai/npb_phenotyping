/**
 * @author Hang Su <hangsu@gatech.edu>.
 */
package edu.gatech.cse8803.clustering

import java.util.Random

import scala.collection.mutable.IndexedSeq

import breeze.linalg.{DenseVector => BreezeVector, DenseMatrix => BreezeMatrix, diag, Transpose}

import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors, DenseVector, DenseMatrix, BLAS}
import org.apache.spark.rdd.RDD

/**
 * @param k The number of independent Gaussians in the mixture model
 * @param convergenceTol The maximum change in log-likelihood at which convergence
 * is considered to have occurred.
 * @param maxIterations The maximum number of iterations to perform
 */
class GaussianMixture  (
                        private var k: Int,
                        private var convergenceTol: Double,
                        private var maxIterations: Int,
                        private var seed: Long) extends Serializable {

  /** A default instance, 2 Gaussians, 100 iterations, 0.01 log-likelihood threshold */
  def this() = this(2, 0.01, 100, new Random().nextLong())

  // number of samples per cluster to use when initializing Gaussians
  private val nSamples = 5

  // an initializing GMM can be provided rather than using the
  // default random starting point
  private var initialModel: Option[GaussianMixtureModel] = None

  /** Set the number of Gaussians in the mixture model.  Default: 2 */
  def setK(k: Int): this.type = {
    this.k = k
    this
  }

  /** Return the number of Gaussians in the mixture model */
  def getK: Int = k

  /** Set the maximum number of iterations to run. Default: 100 */
  def setMaxIterations(maxIterations: Int): this.type = {
    this.maxIterations = maxIterations
    this
  }

  /** Return the maximum number of iterations to run */
  def getMaxIterations: Int = maxIterations

  /**
   * Set the largest change in log-likelihood at which convergence is
   * considered to have occurred.
   */
  def setConvergenceTol(convergenceTol: Double): this.type = {
    this.convergenceTol = convergenceTol
    this
  }

  /**
   * Return the largest change in log-likelihood at which convergence is
   * considered to have occurred.
   */
  def getConvergenceTol: Double = convergenceTol

  /** Set the random seed */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /** Return the random seed */
  def getSeed: Long = seed

  /** Perform expectation maximization */
  def run(data: RDD[Vector]): GaussianMixtureModel = {
    val sc = data.sparkContext

    // we will operate on the data as breeze data
    val breezeData = data.map(u => toBreezeVector(u).toDenseVector).cache()

    // Get length of the input vectors
    val d = breezeData.first().length

    // Determine initial weights and corresponding Gaussians from samples
    val samples = breezeData.takeSample(withReplacement = true, k * nSamples, seed)
    val weights =  Array.fill(k)(1.0 / k)
    val gaussians = Array.tabulate(k) { i =>
          val slice = samples.view(i * nSamples, (i + 1) * nSamples)
          new MultivariateGaussian(vectorMean(slice), initCovariance(slice))
        }

    /**
     * TODO: Implement your code here
     * iterite to train GMM
     */
    new GaussianMixtureModel(weights, gaussians)
  }

  /** Average of dense breeze vectors */
  private def vectorMean(x: IndexedSeq[BreezeVector[Double]]): BreezeVector[Double] = {
    val v = BreezeVector.zeros[Double](x(0).length)
    x.foreach(xi => v += xi)
    v / x.length.toDouble
  }

  /**
   * Construct matrix where diagonal entries are element-wise
   * variance of input vectors (computes biased variance)
   */
  private def initCovariance(x: IndexedSeq[BreezeVector[Double]]): BreezeMatrix[Double] = {
    val mu = vectorMean(x)
    val ss = BreezeVector.zeros[Double](x(0).length)
    x.map(xi => (xi - mu) :^ 2.0).foreach(u => ss += u)
    diag(ss / x.length.toDouble)
  }
}
