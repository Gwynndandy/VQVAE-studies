{
  description = "Python environment for PyTorch script";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
      };
      python = pkgs.python311;
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          (python.withPackages (ps: with ps; [
            numpy
            pytorch
            scipy
            matplotlib
            wandb
          ]))
        ];

        shellHook = ''
          python -c "import numpy, torch, scipy, matplotlib, wandb; print('All imports OK')"
          export WANDB_API_KEY=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
        '';
      };
    };
}
