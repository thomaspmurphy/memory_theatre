defmodule MemoryTheatre.SDM do
  @moduledoc """
  Sparse Distributed Memory (SDM) implementation in Elixir.

  This module implements Kanerva's Sparse Distributed Memory model, which is a mathematical model of human long-term memory.
  The implementation uses Nx for efficient numerical computations.

  ## Key Concepts

  - Address Space: A high-dimensional binary space (typically 1000+ dimensions)
  - Memory Locations: Points in the address space that store data
  - Critical Distance: The distance threshold for activating memory locations
  - Reading/Writing: Operations that use Hamming distance to find and update memory locations

  ## Usage

      iex> sdm = MemoryTheatre.SDM.new(1000, 10000)  # 1000 dimensions, 10000 memory locations
      iex> sdm = MemoryTheatre.SDM.write(sdm, address, data)
      iex> {data, confidence} = MemoryTheatre.SDM.read(sdm, address)
  """

  alias Nx.Tensor, as: Tensor

  @type address :: Tensor.t()
  @type data :: Tensor.t()
  @type memory_location :: {address(), data()}
  @type t :: %__MODULE__{
          dimensions: pos_integer(),
          num_locations: pos_integer(),
          locations: [memory_location()],
          critical_distance: float(),
          critical_distance_factor: float()
        }

  defstruct [:dimensions, :num_locations, :locations, :critical_distance, :critical_distance_factor]

  @default_critical_distance_factor 0.3
  @doc """
  Creates a new SDM instance with the specified number of dimensions and memory locations.

  ## Parameters

    - dimensions: The number of dimensions in the address space (typically 1000+)
    - num_locations: The number of memory locations to create
    - opts: Optional keyword list with :critical_distance_factor

  ## Returns

  A new SDM struct with randomly initialized memory locations.
  """
  def new(dimensions, num_locations, opts \\ []) when dimensions > 0 and num_locations > 0 do
    critical_distance_factor = Keyword.get(opts, :critical_distance_factor, @default_critical_distance_factor)

    # Initialize random memory locations
    locations =
      for _ <- 1..num_locations do
        key = Nx.Random.key(System.unique_integer())
        {random, _} = Nx.Random.uniform(key, 0.0, 1.0, shape: {dimensions}, type: :f32)
        address = Nx.greater(random, 0.5)
        {data, _} = Nx.Random.uniform(Nx.Random.key(System.unique_integer()), 0.0, 1.0, shape: {dimensions}, type: :f32)
        {address, data}
      end

    # Calculate critical distance (typically 0.3 * dimensions, but can be overridden)
    critical_distance = dimensions * critical_distance_factor

    %__MODULE__{
      dimensions: dimensions,
      num_locations: num_locations,
      locations: locations,
      critical_distance: critical_distance,
      critical_distance_factor: critical_distance_factor
    }
  end

  @doc """
  Writes data to the SDM at the specified address.

  ## Parameters

    - sdm: The SDM instance
    - address: The address to write to (binary tensor)
    - data: The data to write (tensor)

  ## Returns

  Updated SDM instance with the new data written to nearby memory locations.
  """
  def write(%__MODULE__{} = sdm, address, data) do
    # Find locations within critical distance
    locations =
      Enum.map(sdm.locations, fn {loc_address, loc_data} ->
        distance = hamming_distance(address, loc_address)
        if distance <= sdm.critical_distance do
          # Update the location's data
          new_data = Nx.add(loc_data, data)
          {loc_address, new_data}
        else
          {loc_address, loc_data}
        end
      end)

    %{sdm | locations: locations}
  end

  @doc """
  Reads data from the SDM at the specified address.

  ## Parameters

    - sdm: The SDM instance
    - address: The address to read from (binary tensor)

  ## Returns

  A tuple containing:
    - The retrieved data
    - A confidence measure (number of locations that contributed to the result)
  """
  def read(%__MODULE__{} = sdm, address) do
    # Find locations within critical distance and sum their data
    {sum_data, count} =
      Enum.reduce(sdm.locations, {Nx.broadcast(0, {sdm.dimensions}), 0}, fn {loc_address, loc_data},
                                                                           {acc_data, acc_count} ->
        distance = hamming_distance(address, loc_address)
        if distance <= sdm.critical_distance do
          {Nx.add(acc_data, loc_data), acc_count + 1}
        else
          {acc_data, acc_count}
        end
      end)

    # Normalize the result
    result =
      if count > 0 do
        Nx.divide(sum_data, count)
      else
        Nx.broadcast(0, {sdm.dimensions})
      end

    {result, count}
  end

  @doc """
  Calculates the Hamming distance between two binary tensors.
  """
  def hamming_distance(tensor1, tensor2) do
    Nx.sum(Nx.not_equal(tensor1, tensor2))
    |> Nx.to_number()
  end
end
