defmodule MemoryTheatre.MixProject do
  use Mix.Project

  def project do
    [
      app: :memory_theatre,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.9.2"},
      {:explorer, "~> 0.10.1"},
      {:ex_doc, "~> 0.38.1", only: :dev, runtime: false}
    ]
  end
end
