import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import cm, colors
positions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
runNumbers = [1234,1231,1235,1233,1236,1237,1238,1239,1240,1241,1243,1245,1246,1248,1249,1250,1252]
beam_coordinates = [25,1032]
table_coordinates = [[25,1032],[25,968],[73,1000],[73,1064],[25,1096],
              [-23,1064],[-23,1000],[-23,936],[25,904],[73,936],
              [121,1000],[121,1064],[73,1128],[25,1160],[-23,1128],
              [-71,1064],[-71,1000]
              ]
x_width = 48
y_width = 64
x_label = "beam position x (mm)"
y_label = "beam position y (mm)"
z_label = r'(#ADC sum - pedestal)/beam E ($GeV^{-1}$)'
mean_response_Cer = []
patches = []

centers = []
x_low_list = []
x_high_list = []
y_low_list = []
y_high_list = []

for run, table_corr in zip(runNumbers, table_coordinates):
    output_json = f"results/root/Run{run}/testbeam_energy_sum_fit.json"
    with open(output_json,"r") as f:
        fit_dic = json.load(f)
        mean_response_Cer.append(fit_dic["mean_response_s_Cer"])
    x_low = beam_coordinates[0] - table_corr[0] - x_width/2
    x_high = beam_coordinates[0] - table_corr[0] + x_width/2
    y_low = beam_coordinates[1] - table_corr[1] - y_width/2
    y_high = beam_coordinates[1] - table_corr[1] + y_width/2
    x_low_list.append(x_low)
    x_high_list.append(x_high)
    y_low_list.append(y_low)
    y_high_list.append(y_high)    
    rect = Rectangle((x_low, y_low), x_high - x_low, y_high - y_low)
    patches.append(rect)
    centers.append((beam_coordinates[0] - table_corr[0], beam_coordinates[1] - table_corr[1]))
fig, ax = plt.subplots(figsize=(6, 5))
collection = PatchCollection(patches, cmap="viridis", edgecolor="k", linewidth=0.5)
collection.set_array(mean_response_Cer)
ax.add_collection(collection)
# Colormap setup
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(vmin=min(mean_response_Cer), vmax=max(mean_response_Cer))
for pos, res, center in zip(positions,mean_response_Cer,centers):
    text = f"#{pos}:\n {res:.2f}"
    rgba = cmap(norm(res))
    # compute perceived brightness (YIQ formula)
    brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    text_color = "black" if brightness > 0.5 else "white"
    ax.text(center[0], center[1], text, ha="center", va="center", fontsize=10, color=text_color)

# Set axis limits
ax.set_xlim(min(x_low_list), max(x_high_list))
ax.set_ylim(min(y_low_list), max(y_high_list))
ax.set_aspect("equal", adjustable="box")
cbar = fig.colorbar(collection, ax=ax, label=z_label)
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_title("FERS average response (Cer)")
plt.savefig("mean_response_vs_position.png")

