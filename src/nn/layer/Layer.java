package nn.layer;

public class Layer {

	private final Neuron[] ns, grads;

	private final double[] inputs;
	private final double[] weightedInputs;
	private double[] activations;

	private final static double LEAKY_RELU_X = 0.01;

	public Layer(final int numInputs, final int numOutputs)
	{
		inputs = new double[numOutputs];
		weightedInputs = new double[numOutputs];
		ns = Neuron.createNeurons(numOutputs, numInputs);
		grads = Neuron.createNeurons(numOutputs, numInputs);
		Neuron.randomizeWeights(ns);
	}

	public double[] calcOutput(final double[] inputs)
	{
		for( int i = 0; i < inputs.length; ++i ) {
			this.inputs[i] = inputs[i];
		}

		final double[] activations = new double[ns.length];

		for( int i = 0; i < ns.length; ++i ) {
			final Neuron n = ns[i];
			final double weightedInput = n.sqrtSum(inputs);

			//Activation
			weightedInputs[i] = weightedInput;
			activations[i] = calcReLU(weightedInput);
		}

		this.activations = activations;
		return activations;
	}

	public void applyGradients(final double learnRate)
	{
		for( int i = 0; i < ns.length; ++i ) {
			ns[i].applyGradient(grads[i], learnRate);
		}
	}

	public static double calcReLU(final double weightedInput)
	{
		//Leaky ReLU
		return Math.max(LEAKY_RELU_X * weightedInput, weightedInput);
		//return 1 / (1 + Math.exp(-weightedInput));
	}

	public static double calcReLUDerivative(final double weightedInput)
	{
		return weightedInput < 0 ? LEAKY_RELU_X : 1;

		//double sig = calcReLU(weightedInput);
		//return sig * (1 - sig);
	}

	public double calcLayerCost(final double outputGiven, final double outputExpected)
	{
		return Math.pow(outputExpected - outputGiven, 2);
	}

	public double calcLayerCostDerivative(final double outputActivation, final double expectedOutput)
	{
		return 2 * (outputActivation - expectedOutput);
	}

	public double[] calcOutputLayerNodeValues(final double[] expectedOutputs)
	{
		//System.out.println("d " + Arrays.toString(activations));
		//System.out.println("c " + Arrays.toString(weightedInputs));
		final double[] nodeValues = new double[expectedOutputs.length];

		for( int i = 0; i < nodeValues.length; ++i ) {
			final double dCost = calcLayerCostDerivative(activations[i], expectedOutputs[i]);
			final double dActivation = calcReLUDerivative(weightedInputs[i]);
			nodeValues[i] = dActivation * dCost;
		}

		//System.out.println(Arrays.toString(nodeValues));

		return nodeValues;
	}

	public void updateGradients(final double[] nodeValues)
	{
		for( int i = 0; i < grads.length; ++i ) {
			grads[i].updateGradient(inputs, nodeValues[i]);
		}
		//System.out.println(Arrays.deepToString(weightGradients));
	}

	public double[] calcHiddenLayerNodeValues(final Layer oldLayer, final double[] oldNodeValues)
	{
		final double[] newNodeValues = new double[ns.length];
		for( int i = 0; i < ns.length; ++i ) {
			final Neuron oldn = oldLayer.ns[i];
			double newNodeValue = 0;
			for( int j = 0; j < oldNodeValues.length; ++j ) {
				newNodeValue += oldn.weights[j] * oldNodeValues[j];
			}
			newNodeValue *= calcReLUDerivative(weightedInputs[i]);
			newNodeValues[i] = newNodeValue;
		}

		return newNodeValues;
	}

	public void clearGradients() {
		for( int i = 0; i < grads.length; ++i ) {
			grads[i].clear();
		}
	}
}
