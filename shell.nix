{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "pipzone";
  targetPkgs = pkgs: (with pkgs; [
    just
    python311
    python311Packages.pip
    python311Packages.virtualenv
    zlib
  ]);
  runScript = "bash";
}).env
