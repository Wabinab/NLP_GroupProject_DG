# How to use Git (basics)
`git clone ...`  
`git checkout -b new_branch_name` (checkout to a new branch that may or may not exist).  
... make your changes...  
`git add .` (or `git add -u` if not all you want to add)  
`git commit -m "any message"`  
`git push` (and copy the error code if it want to push to upstream and paste it. This is when you first define a new branch).   

## What if you forked it and commit there?
This assumes you have your thing on your computer. And you add a commit. So this is what you do. 
- `git add .` (I assumed you do this)
- `git commit -am "any message"` (**remember to add the `-am` rather than `-m`**).
- `gh pr create -B main -t "enter title" -b "enter body of PR here"` to create pull request. 

## Rules:
- **Never push to main**/master. If we make mistake, we can delete a side branch. However if otherwise, we may have to delete the whole repo. 
- *Pull from master* (and push to your own development branch). You're allowed to make a new branch. 
- Consider having each of us checking each of us's work especially for work that requires combining into a final work, if we're doing the same thing. (This doesn't mean we don't retain our original work. In fact, we can have 3 notebooks for example, one by me, one by you, and one combined, and we just need the combined one to hand in if requested). **Hence, open pull requests**. 
- Check **Projects** section in github (this is the code section, there're other sections like Issues, Pull Requests): there's a Kanban project where you can pull the required cards to the correct status. Note the Kanban have some automation on moving Pull Requests object (but not manually created cards). 
- (Optional) Check Wiki section if you want to add a page and describe some stuffs you made; or consider it an update log. 
- I set up branch protection rules so every single push requires review. **We'll see how annoying that goes** before I decide whether to remove that or not. Still, I blocked pushing directly to main. 
- Any questions? 

## Colab:
You could also directly push from Colab to the github repo. Just click `File --> Save a copy in github` and find this repo: `Wabinab/NLP_GroupProject_DG` and **make sure you already have a side branch made here before pushing or it'll failed since I blocked pushing directly to main**. 
