# TODO

- docs
- better and more tests
- lots of comments in files
- use lmfit, pytorch, tensorflow, deeptime, etc for inspiration on architecture and philosophy
  - use nn's to get a sense of how and why and usecases for this
- more precise reqs
- Subspecies and better scheme generation so that i dont have to make entirely new copied transitions for modified species, like L and Lsil.
  - maybe subspecies can share transitions?
  - class based
    - identifies missing params and fits them based on data
- Kinetic scheme where there is a continuum of species over a colvar? instead of a continuum of population. and a third party affects the pop dist of this species space which affects the binding, like if all species share a common transition like kon.
  - like NA lecture 10 RecA
  - put in prob dist
  - explored in kinetics6.ipynb in kinetic_modeling
- plotting utils:
  - interactive plot
- allow symbolic rate constants in case they are not known
