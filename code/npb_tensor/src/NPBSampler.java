import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;
import java.util.Scanner;
import java.util.TreeSet;

class Record {   
	
	int patIndex;
	int medIndex;
	int diagIndex;
	
	public Record(int patIndex, int medIndex, int diagIndex){
		this.patIndex = patIndex;
		this.medIndex = medIndex;
		this.diagIndex = diagIndex;
	}
}

class TensorHolder{
	public Record[] records;
	public int numPat;
	public int numMed;
	public int numDiag;
	public int numRecords;
	
	public TensorHolder(InputStream is) throws IOException {
		Scanner sc = new Scanner(is);
		int length = sc.nextInt();
		records = new Record[length];
		numPat = sc.nextInt();
		numMed = sc.nextInt();
		numDiag = sc.nextInt();
		for (int i = 0; i < length; ++i)
		{
			int x = sc.nextInt();
			int y = sc.nextInt(); 
			int z = sc.nextInt();
			records[i] = new Record(x, y, z);
		}
		numRecords = length;
		sc.close();
	}
}

public class NPBSampler
{
	public int K; // number of phenotypes;
	public ArrayList<Integer> kactive;
	public TreeSet<Integer> kgaps;
	
	public ArrayList<int[]> numPhenoPat;
	public ArrayList<int[]> numPhenoMed;
	public ArrayList<int[]> numPhenoDiag;
	public ArrayList<Integer> numPheno;
	public int[] z;
	public TensorHolder ts;
	private double[] pp;
	
	private double alpha = 0.5;
	private double gamma_base = 50;
	private double gamma_p;
	private double gamma_m;
	private double gamma_d;
	
	private Random rand;
	
	private void initObjects()
	{		
		rand = new Random();

		numPhenoPat = new ArrayList<int[]>();
		numPhenoMed = new ArrayList<int[]>();
		numPhenoDiag = new ArrayList<int[]>();
		numPheno = new ArrayList<Integer>();
		for (int k = 0; k < K; k++) {
			numPheno.add(0);
		}
		
		kactive = new ArrayList<Integer>();
		kgaps = new TreeSet<Integer>();
		
		for (int k = 0; k < K; k++) 
			kactive.add(k);
		
		pp = new double[K + 10];
	}
	
	private void doShuffle()
	{
		for (int i = 1; i < ts.numRecords; ++i)
		{
			int j = rand.nextInt(i);
			Record tmp = ts.records[i];
			ts.records[i] = ts.records[j];
			ts.records[j] = tmp;
			int tk = z[i];
			z[i] = z[j];
			z[j] = tk;
		}
	}
	
	public void initRandom()
	{
		z = new int[ts.numRecords];
		
		for (int k = 0; k < K; k++)
		{
			numPhenoPat.add(new int[ts.numPat]);
			numPhenoMed.add(new int[ts.numMed]);
			numPhenoDiag.add(new int[ts.numDiag]);
		}
		
		for (int i = 0; i < z.length; ++i)
		{
			z[i] = rand.nextInt(K);
			numPhenoPat.get(z[i])[ts.records[i].patIndex]++;
			numPhenoMed.get(z[i])[ts.records[i].medIndex]++;
			numPhenoDiag.get(z[i])[ts.records[i].diagIndex]++;
			numPheno.set(z[i], numPheno.get(z[i]) + 1);
		}
	}
	
	private int spawnPhenotype(int n, int pat, int med, int diag) {
		int k;
		if (kgaps.size() > 0) {
			// reuse gap
			// k = kgaps.remove(kgaps.size() - 1);
			k = kgaps.first();
			kgaps.remove(k);
			kactive.add(k);
			numPheno.set(k, 1);
		} else {
			// add element to count arrays
			k = K;
			kactive.add(K);
			numPhenoPat.add(new int[ts.numPat]);
			numPhenoMed.add(new int[ts.numMed]);
			numPhenoDiag.add(new int[ts.numDiag]);
			numPheno.add(1);
		}
		numPhenoPat.get(k)[pat] = 1;
		numPhenoMed.get(k)[med] = 1;
		numPhenoDiag.get(k)[diag] = 1;
		K++;
		if (pp.length <= K) {
			pp = new double[K + 10];
		}
		return k;
	}
	
	public void nextGibbsSweep()
	{
		for (int n = 0; n < ts.numRecords; ++n)
		{
			int k = z[n];
			numPheno.set(k, numPheno.get(k) - 1);
			int pat = ts.records[n].patIndex;
			int med = ts.records[n].medIndex;
			int diag = ts.records[n].diagIndex;
			numPhenoPat.get(k)[pat]--;
			numPhenoMed.get(k)[med]--;
			numPhenoDiag.get(k)[diag]--;
			int kold = k;
			
			// compute weights
			double psum = 0;
			// (37)
			for (int kk = 0; kk < K; kk++) {
				k = kactive.get(kk);
				pp[kk] = numPheno.get(k) * 
						(numPhenoPat.get(k)[pat] + gamma_p) / (numPheno.get(k) + ts.numPat * gamma_p) * 
						(numPhenoMed.get(k)[med] + gamma_m) / (numPheno.get(k) + ts.numMed * gamma_m) * 
						(numPhenoDiag.get(k)[diag] + gamma_d) / (numPheno.get(k) + ts.numDiag * gamma_d);
						
				psum += pp[kk];
			}
			// likelihood of new component
			pp[K] = alpha / ts.numPat / ts.numDiag / ts.numMed;
			psum += pp[K];
			
			double u = rand.nextDouble();
			u *= psum;
			psum = 0;
			int kk = 0;
			for (; kk < K + 1; kk++) {
				psum += pp[kk];
				if (u <= psum) {
					break;
				}
			}
			
			// reassign and increment
			if (kk < K) 
			{
				k = kactive.get(kk);
				z[n] = k;
				numPheno.set(k, numPheno.get(k) + 1);
				numPhenoPat.get(k)[pat]++;
				numPhenoMed.get(k)[med]++;
				numPhenoDiag.get(k)[diag]++;
			} else {
				z[n] = spawnPhenotype(n, pat, med, diag);
				System.out.println("K = " + K);
			}
			
			// empty topic?
			if (numPheno.get(kold) == 0) 
			{
				// remove the object not the index
				kactive.remove((Integer) kold);
				kgaps.add(kold);
				K--;
				System.out.println("K = " + K);
			}
		}
	}
	
	public void packTopics() {
	    Iterator<Integer> iterator = kgaps.descendingIterator();
	    int k;
	    while (iterator.hasNext())
	    {
	    	k = iterator.next();
			numPhenoPat.remove(k);
			numPhenoMed.remove(k);
			numPhenoDiag.remove(k);
			numPheno.remove(k);
	    }
		kgaps.clear();
		kactive.clear();
		for (int i = 0; i < K; ++i)
			kactive.add(i);
	}
	
	public void run(int maxIter)
	{
		for (int iter = 0; iter < maxIter; ++iter)
		{
			doShuffle();
			nextGibbsSweep();
			System.out.println("iter = " + iter + " #phenotypes = " + kactive.size());
		}
		packTopics();
	}

	public NPBSampler(TensorHolder ts, int K0)
	{
		this.ts = ts;
		K = K0;
		gamma_p = gamma_base / ts.numPat;
		gamma_m = gamma_base / ts.numMed;
		gamma_d = gamma_base / ts.numDiag;
		initObjects();
		initRandom();
	}
	
	public static void printMedDistribution(NPBSampler sampler, Integer k) throws IOException
	{
		PrintWriter writer = new PrintWriter("med-" + k.toString() + ".txt", "UTF-8");
		Double base = sampler.gamma_m * sampler.ts.numMed + sampler.numPheno.get(k);
		for (int i = 0; i < sampler.ts.numMed; ++i)
		{
			Double prob = (sampler.gamma_m + sampler.numPhenoMed.get(k)[i]) / base; 
			writer.write(prob.toString() + "\n");
		}
		writer.close();
	}
	
	public static void printDiagDistribution(NPBSampler sampler, Integer k) throws IOException
	{
		PrintWriter writer = new PrintWriter("diag-" + k.toString() + ".txt", "UTF-8");
		Double base = sampler.gamma_d * sampler.ts.numDiag + sampler.numPheno.get(k);
		for (int i = 0; i < sampler.ts.numDiag; ++i)
		{
			Double prob = (sampler.gamma_d + sampler.numPhenoDiag.get(k)[i] / base);
			writer.write(prob.toString() + "\n");
		}
		writer.close();
	}
	
	public static void main(String[] args) throws IOException {
		TensorHolder ts = new TensorHolder(new FileInputStream("mimic2-triplets.txt"));
		NPBSampler sampler = new NPBSampler(ts, 10);
		sampler.run(1000);
		for (int k = 0; k < sampler.K; ++k)
			printMedDistribution(sampler, k);
		
		for (int k = 0; k < sampler.K; ++k)
			printDiagDistribution(sampler, k);
	}
}