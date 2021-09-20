
# ! download 
localdir=/cygdrive/c/Users/duongdb/Documents/FH_OCT_08172021/Stylegan2
mkdir $localdir
maindir=/data/duongdb/FH_OCT_08172021/Stylegan2
for mod in 00000-Tf256RmFold3+EyePos+FH-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel4
do 
mkdir $localdir/$mod
scp duongdb@helix.nih.gov:$maindir/$mod/*png $localdir/$mod
done

