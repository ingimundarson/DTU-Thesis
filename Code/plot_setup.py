import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# custom mpl
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams["figure.figsize"] = (10,4)
mpl.rc('lines', markersize = 10, linewidth = 3)
mpl.rc('axes', labelsize = 14, titlesize = 20)

# custom sns
sns.set_style("whitegrid")

# get dtu colors
from matplotlib.colors import LinearSegmentedColormap


colors = {
    'DTU Red': '#990000',
    'DTU Blue': '#2f3eea',
    'DTU Green': '#1fd082',
    'DTU dBlue': '#030f4f',
    'DTU Yellow': '#f6d04d',
    'DTU Orange': '#fc7634',
    'DTU Pink': '#f7bbb1',
    'DTU Gray': '#dadada',
    'DTU dPink': '#e83f48',
    'DTU dGreen': '#008835',
    'DTU Purple': '#79238e'
}

dtu_colors_rgb = []
for name, c in colors.items():
    if name == "DTU Gray":
        continue
        
    dtu_colors_rgb.append(mpl.colors.to_rgb(c))
    
dtu_colors_hex = []
for c in dtu_colors_rgb:
    dtu_colors_hex.append(mpl.colors.to_hex(c))
    
dtu_cmap = LinearSegmentedColormap.from_list("dtu_colors", dtu_colors_rgb, N=10)
dtu_cmap_medium = LinearSegmentedColormap.from_list("dtu_colors_medium", dtu_colors_rgb, N=15)
dtu_cmap_large = LinearSegmentedColormap.from_list("dtu_colors_large", dtu_colors_rgb, N=25)
dtu_cmap_continuous = LinearSegmentedColormap.from_list("dtu_colors_continuous", [dtu_cmap.get_bad(), dtu_cmap.get_under(), dtu_cmap.get_bad()], N=256 )

mpl.colormaps.unregister("dtu_colors")
mpl.colormaps.unregister("dtu_colors_medium")
mpl.colormaps.unregister("dtu_colors_large")
mpl.colormaps.unregister("dtu_colors_continuous")

mpl.colormaps.register(dtu_cmap)
mpl.colormaps.register(dtu_cmap_medium)
mpl.colormaps.register(dtu_cmap_large)
mpl.colormaps.register(dtu_cmap_continuous)
dtu_sns_palette = sns.color_palette(dtu_colors_hex)

# set color maps
# sns_palette = sns.color_palette("deep")
# cmap = ListedColormap(sns_palette)

mpl.rc('image', cmap="dtu_colors")
sns.set_palette(dtu_sns_palette)


