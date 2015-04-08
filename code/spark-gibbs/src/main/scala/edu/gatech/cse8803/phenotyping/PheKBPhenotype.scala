/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse8803.phenotyping

import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import org.apache.spark.rdd.RDD

object T2dmPhenotype {
  /**
   * Transform given data set to a RDD of patients and corresponding phenotype
   * @param medication medication RDD
   * @param labResult lab result RDD
   * @param diagnostic diagnostic code RDD
   * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
   */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {

    /**
     * Remove the place holder and implement your code here,
     * hard code the medication, lab, icd code etc for
     * phenotype as while testing your code we expect
     * your function have no side effect, i.e. don't
     * read from file or write file
     */
    medication.sparkContext.parallelize(Seq(("patient-one", 1), ("patient-two", 0)))
  }
}
