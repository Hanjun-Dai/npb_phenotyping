import AssemblyKeys._

assemblySettings

name := "big-data-hw2"

version := "1.0"

scalaVersion := "2.10.4"

resolvers ++= Seq(
  Resolver.sonatypeRepo("releases"),
  Resolver.sonatypeRepo("snapshots")
)

libraryDependencies ++= Seq(
  "org.apache.spark"  % "spark-core_2.10"              % "1.2.0" % "provided",
  "org.apache.spark"  % "spark-mllib_2.10"             % "1.2.0",
  "com.databricks"    % "spark-csv_2.10"               % "0.1",
  "com.chuusai"       % "shapeless_2.10.4"             % "2.0.0"
)

mainClass in assembly := Some("edu.gatech.cse8803.main.Main")

mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
{
  case m if m.toLowerCase.endsWith("manifest.mf") => MergeStrategy.discard
  case m if m.startsWith("META-INF") => MergeStrategy.discard
  case PathList("javax", "servlet", xs @ _*) => MergeStrategy.first
  case PathList("org", "apache", xs @ _*) => MergeStrategy.first
  case PathList("org", "jboss", xs @ _*) => MergeStrategy.first
  case "about.html"  => MergeStrategy.rename
  case "reference.conf" => MergeStrategy.concat
  case _ => MergeStrategy.first
}
}
