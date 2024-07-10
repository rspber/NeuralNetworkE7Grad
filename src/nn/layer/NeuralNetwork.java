package nn.layer;

public class NeuralNetwork {

	final Layer[] layers;

	public NeuralNetwork(final int[] layerSizes)
	{
		layers = new Layer[layerSizes.length - 1];
		for( int i = 0; i < layers.length; ++i ) {
			layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
		}
	}

	/*
	 * calcAllLayersOutput
	 */
	public double[] calcAllLayersOutput(double[] inputs) {
		for( final Layer layer : layers ) {
			inputs = layer.calcOutput(inputs);
		}
		return inputs;
	}

	public int getClassification(final double[] inputs)
	{
		final double[] outputs = calcAllLayersOutput(inputs);

		int indexOfMax = 0;
		double max = outputs[indexOfMax];
		for( int i = 1; i < outputs.length; ++i ) {
			if( outputs[i] > max ) {
				max = outputs[i];
				indexOfMax = i;
			}
		}
		return indexOfMax;
	}

	public double calcSingleCost(final DataPoint dataPoint)
	{
		final double[] outputs = calcAllLayersOutput(dataPoint.inputs);
		final Layer outputLayer = layers[layers.length - 1];
		double cost = 0;

		for( int i = 0; i < outputs.length; ++i ) {
			cost += outputLayer.calcLayerCost(outputs[i], dataPoint.expectedOutputs[i]);
		}
		
		return cost;
	}

	public double calcMultipleCost(final DataPoint[] data)
	{
		double totalCost = 0;

		for( final DataPoint dataPoint : data ) {
			totalCost += calcSingleCost(dataPoint);
		}

		return totalCost / data.length;
	}

	public void applyAllGradients(final double learnRate)
	{
		for( final Layer layer : layers ) {
			layer.applyGradients(learnRate);
		}
	}

	public void updateAllGradients(final DataPoint dataPoint)
	{
		calcAllLayersOutput(dataPoint.inputs);

		final Layer outputLayer = layers[layers.length - 1];
		double[] nodeValues = outputLayer.calcOutputLayerNodeValues(dataPoint.expectedOutputs);

		outputLayer.updateGradients(nodeValues);

		for( int i = layers.length - 1; --i >= 0; ) {
			final Layer hiddenLayer = layers[i];
			nodeValues = hiddenLayer.calcHiddenLayerNodeValues(layers[i + 1], nodeValues);

			hiddenLayer.updateGradients(nodeValues);
		}
	}

	public void clearAllGradients() {
		for( final Layer layer : layers ) {
			layer.clearGradients();
		}
	}

	public void learn(final DataPoint[] trainingData, final double learnRate)
	{
		for( final DataPoint datapoint : trainingData ) {
			updateAllGradients(datapoint);
		}
		applyAllGradients( learnRate / trainingData.length );
		clearAllGradients();
	}

}
