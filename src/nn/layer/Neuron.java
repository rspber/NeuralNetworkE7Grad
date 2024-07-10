package nn.layer;

import java.util.Arrays;

public class Neuron {

	final double[] weights;
	double bias;

	public Neuron(final int numInputs)
	{
		weights = new double[numInputs];
	}


	public void clear()
	{
		Arrays.fill(weights, 0);
	}

	public double sqrtSum(final double[] inputs)
	{
		double sqrtSum = bias;
		for (int i = 0; i < inputs.length; i++) {
			sqrtSum += inputs[i] * weights[i];
		}
		return sqrtSum;
	}

	private double randomInNormalDistribution(final double mean, final double standardDeviation)
	{
		final double x1 = 1 - Math.random();
		final double x2 = 1 - Math.random();

		final double y1 = Math.sqrt(-2.0 * Math.log(x1)) * Math.cos(2.0 * Math.PI * x2);
		return y1 * standardDeviation + mean;
	}

	public void randomizeWeights()
	{
		for( int i = 0; i < weights.length; ++i ) {
			weights[i] = randomInNormalDistribution(0, 1) / Math.sqrt(weights.length);
		}
	}

	public void updateGradient(final double[] inputs, final double nodeValue)
	{
		for( int i = 0; i < weights.length; ++i ) {
			weights[i] += inputs[i] * nodeValue;
		}
		bias += nodeValue;
	}

	public void applyGradient(final Neuron grad, final double learnRate)
	{
		bias -= grad.bias * learnRate;
		for( int i = 0; i < weights.length; ++i ) {
			weights[i] -= grad.weights[i] * learnRate;
		}
	}

	public static Neuron[] createNeurons(final int numOut, final int numIn)
	{
		Neuron[] ns = new Neuron[numOut];
		for( int i = 0; i < ns.length; ++i ) {
			ns[i] = new Neuron(numIn);
		}
		return ns;
	}

	public static void randomizeWeights(final Neuron[] ns)
	{
		for( int i = 0; i < ns.length; ++i ) {
			ns[i].randomizeWeights();
		}
	}

	public static void clearNeurons(final Neuron[] ns)
	{
		for( int i = 0; i < ns.length; ++i ) {
			ns[i].clear();
		}
	}

}
