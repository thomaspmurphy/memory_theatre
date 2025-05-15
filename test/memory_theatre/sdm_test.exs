defmodule MemoryTheatre.SDMTest do
  use ExUnit.Case
  alias MemoryTheatre.SDM

  describe "SDM" do
    test "creates a new SDM with correct dimensions and locations" do
      dimensions = 100
      num_locations = 1000
      sdm = SDM.new(dimensions, num_locations)

      assert sdm.dimensions == dimensions
      assert sdm.num_locations == num_locations
      assert length(sdm.locations) == num_locations
      assert sdm.critical_distance == dimensions * 0.3

      # Verify each location has correct dimensions
      Enum.each(sdm.locations, fn {address, data} ->
        assert Nx.shape(address) == {dimensions}
        assert Nx.shape(data) == {dimensions}
        assert Nx.type(address) == {:u, 8}  # Nx.greater returns {:u, 8} by default
        assert Nx.type(data) == {:f, 32}    # Float tensor (f32)
      end)
    end

    test "writes and reads data correctly" do
      sdm = SDM.new(100, 1000, critical_distance_factor: 0.4)  # Use a slightly higher critical distance factor for testing

      # Create test address and data using Nx.Random (as in SDM.new) so that the address is generated in the same way as the memory locations.
      key = Nx.Random.key(System.unique_integer())
      {random, _} = Nx.Random.uniform(key, 0.0, 1.0, shape: {100}, type: :f32)
      address = Nx.greater(random, 0.5)
      {data, _} = Nx.Random.uniform(Nx.Random.key(System.unique_integer()), 0.0, 1.0, shape: {100}, type: :f32)

      # Write data
      sdm = SDM.write(sdm, address, data)

      # Read data back
      {retrieved_data, confidence} = SDM.read(sdm, address)

      # Verify we got some confidence (some locations were activated)
      assert confidence > 0

      # Verify the retrieved data has the correct shape
      assert Nx.shape(retrieved_data) == {100}
      assert Nx.type(retrieved_data) == {:f, 32}
    end

    test "handles empty reads gracefully" do
      sdm = SDM.new(100, 1000)

      # Create an address that's very different from any location
      # (all ones, which should be far from random addresses)
      address = Nx.broadcast(1, {100})

      {retrieved_data, confidence} = SDM.read(sdm, address)

      # Should get zero confidence and zero data
      assert confidence == 0
      assert Nx.to_number(Nx.sum(retrieved_data)) == 0
    end

    test "hamming distance calculation" do
      # Test with known values
      t1 = Nx.tensor([1, 0, 1, 0, 1], type: {:u, 8})
      t2 = Nx.tensor([1, 1, 0, 0, 1], type: {:u, 8})

      # Should be 2 bits different
      assert SDM.hamming_distance(t1, t2) == 2

      # Same tensor should have distance 0
      assert SDM.hamming_distance(t1, t1) == 0
    end
  end
end
