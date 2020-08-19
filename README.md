# TAA-STAR

Implementations of several state of the art Temporal Anti-Aliasing (TAA) techniques.

## Project Description

TBD.

## Setup Instructions and Submodules

1. Clone the repository with all submodules recursively:       
`git clone https://github.com/cg-tuwien/TAA-STAR.git . --recursive` (to check out into `.`)
2. Open `taa.sln` with Visual Studio 2019, set `taa` as the startup project, build and run

You might want to have all submodules checked-out to their `master` branch. You can do so using:      
`git submodule foreach --recursive git checkout master`.       
There are two submodules: One under `gears_vk/` (referencing https://github.com/cg-tuwien/Gears-Vk) and another under `gears_vk/auto_vk/` (referencing https://github.com/cg-tuwien/Auto-Vk).    

To update the submodules on a daily basis, use one of the following commands:
* `git submodule update --recursive`
* `git submodule foreach "(git checkout master; git pull)&"`
* TBD which one is the most practicable (credits: [StackOverflow question](https://stackoverflow.com/questions/1030169/easy-way-to-pull-latest-of-all-git-submodules))

To contribute to either of the submodules, please do so via pull requests and follow the ["Contributing Guidelines" from Gears-Vk](https://github.com/cg-tuwien/Gears-Vk/blob/master/CONTRIBUTING.md). Every time you check something in, make sure that the correct submodule-commits (may also reference forks) are referenced so that one can always get a compiling and working version by cloning as described in step 1!

## Documentation 

TBD.

## License 

TBD.
