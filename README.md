# Memory Theatre

![image](https://github.com/user-attachments/assets/e87bdd4a-6334-4dd1-a85d-f3253fa50f24)

Memory Theatre implements **Sparse Distributed Memory (SDM)**, a mathematical model of associative memory inspired by Kanerva's work. SDM is designed for high-dimensional binary spaces and is capable of storing and retrieving patterns even in the presence of noise.

## Mathematics of Sparse Distributed Memory

**Address Space:**
SDM operates in a high-dimensional binary space (e.g., 1000+ dimensions). Each memory location is a random binary vector (address) in this space.

**Critical Distance:**
When reading or writing, SDM activates all memory locations within a certain Hamming distance (the number of differing bits) from the query address. The critical distance $d_c$ is typically set as a fraction of the number of dimensions:

$$
d_c = \alpha \cdot n
$$

where $n$ is the number of dimensions and $\alpha$ is a tunable parameter (e.g., 0.3 or 0.4).

**Write Operation:**
To write a data vector $x$ at address $a$:
1. Find all memory locations whose address is within $d_c$ of $a$.
2. For each such location, add $x$ to its stored data vector.

**Read Operation:**
To read from address $a$:
1. Find all memory locations within $d_c$ of $a$.
2. Sum their data vectors and (optionally) normalize by the number of contributors.

**Hamming Distance:**
Given two binary vectors $a, b$, the Hamming distance is:

$$
H(a, b) = \sum_{i=1}^n |a_i - b_i|
$$

## Usage Examples (in IEx)

Start an IEx session:

```elixir
iex -S mix
```

Create a new SDM with 100 dimensions and 1000 memory locations:

```elixir
iex> sdm = MemoryTheatre.SDM.new(100, 1000)
%MemoryTheatre.SDM{...}
```

Generate a random binary address and a random data vector:

```elixir
iex> key = Nx.Random.key(System.unique_integer())
iex> {rand, _} = Nx.Random.uniform(key, 0.0, 1.0, shape: {100}, type: :f32)
iex> address = Nx.greater(rand, 0.5)
iex> {data, _} = Nx.Random.uniform(Nx.Random.key(System.unique_integer()), 0.0, 1.0, shape: {100}, type: :f32)
```

Write the data to the SDM:

```elixir
iex> sdm = MemoryTheatre.SDM.write(sdm, address, data)
%MemoryTheatre.SDM{...}
```

Read the data back:

```elixir
iex> {retrieved, confidence} = MemoryTheatre.SDM.read(sdm, address)
iex> confidence
# Number of locations that contributed to the read
iex> Nx.shape(retrieved)
{100}
```

Check Hamming distance between two addresses:

```elixir
iex> a = Nx.tensor([1, 0, 1, 0, 1], type: {:u, 8})
iex> b = Nx.tensor([1, 1, 0, 0, 1], type: {:u, 8})
iex> MemoryTheatre.SDM.hamming_distance(a, b)
2
```

## Installation

<!-- If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `memory_theatre` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:memory_theatre, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/memory_theatre>.
 -->
