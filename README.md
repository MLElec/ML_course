# EPFL Machine Learning Course CS-433
Machine Learning Course, Fall 2017

Repository for all lecture notes, labs and projects - resources, code templates and solutions.

The course website and syllabus is available here: [https://mlo.epfl.ch/page-146520.html]

Contact us if you have any questions: [epfmlcourse@gmail.com](mailto:epfmlcourse@gmail.com), or feel free to create issues and pull requests using the menu above.


## Configure remote branch

You must configure a remote that points to the upstream repository in Git to sync changes you make in a fork with the original repository. This also allows you to sync changes made in the original repository with the fork.

```
git remote -v
git remote add upstream https://github.com/epfml/ML_course.git
git remote -v
```

## Update remote branch
    
Sync a fork of a repository to keep it up-to-date with the upstream repository.

Before you can sync your fork with an upstream repository, you must configure a remote that points to the upstream repository in Git.

```
git fetch upstream
git checkout master
git merge upstream/master
```

Original post: [Syncing a fork](https://help.github.com/articles/syncing-a-fork/)
