package nn.layer;

public class DataPoint {

	final double[] inputs;
	final double[] expectedOutputs;

	public DataPoint(final double[] inputs, final double[] expectedOutputs)
	{
		this.inputs = inputs;
		this.expectedOutputs = expectedOutputs;
	}

	public DataPoint (final double[] inputs, final int expectedOutput)
	{
		this.inputs = inputs;
		this.expectedOutputs = new double[10];
		this.expectedOutputs[expectedOutput] = 1;
	}

}
