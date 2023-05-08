import numpy as np
import torch
import dgl


def _send_color(edges):
  return {'color': edges.src['color']}


def _gen_create_multiset(num_nodes):
  def _create_multiset(nodes):
    end = nodes.mailbox['color'].shape[1]
    multiset = torch.zeros((nodes.batch_size(), num_nodes)) - 1
    multiset[:, 0] = nodes.data['color']
    multiset[:, 1:end + 1] = nodes.mailbox['color'].sort().values
    return {'color': multiset}
  return _create_multiset


def _to_color(colors):
  colors = colors[colors >= 0]
  self_color = colors.astype(int).astype(str)[0]
  neighbour_color = colors[1:].astype(int).astype(str).tolist()
  return self_color + '|' + ','.join(neighbour_color)


def _update_colors(G):
  # G.update_all()
  return list(map(_to_color, G.ndata.pop('color').cpu().numpy()))


def is_possibly_isomorphic(G, H, max_iter=10):
  """Check if the two given graphs are possibly isomorphic by the 1-Weisfeiler-Lehman algorithm.

  Arguments:
      G {networkx.classes.graph.Graph} -- Graph
      H {networkx.classes.graph.Graph} -- Graph

  Keyword Arguments:
      max_iter {int} -- The max number of iterations(default: {10})

  Returns:
      bool -- Return "False" if definitely not isomorphic
  """
  G = dgl.DGLGraph(G)
  H = dgl.DGLGraph(H)

  # If the number of nodes is different, return False.
  if G.number_of_nodes() != H.number_of_nodes():
    return False

  # Set initial colors
  G.ndata['color'] = np.zeros(G.number_of_nodes())
  H.ndata['color'] = np.zeros(H.number_of_nodes())
  N = G.number_of_dst_nodes()
  current_max_color = 0

  G.register_message_func(_send_color)
  G.register_reduce_func(_gen_create_multiset(N))
  H.register_message_func(_send_color)
  H.register_reduce_func(_gen_create_multiset(N))

  # Refine colors until convergence
  for i in range(max_iter):
    G_colors = _update_colors(G)
    H_colors = _update_colors(H)

    G_unique_colors, G_counts = np.unique(G_colors, return_counts=True)
    H_unique_colors, H_counts = np.unique(H_colors, return_counts=True)
    G_multiset = dict(zip(G_unique_colors, G_counts))
    H_multiset = dict(zip(H_unique_colors, H_counts))

    if G_multiset != H_multiset:
      return False

    # Recoloring (str -> int)
    unique_colors = np.unique(np.append(G_unique_colors, H_unique_colors))
    recolor_map = {color: i + 1 for i, color in enumerate(unique_colors)}

    G.ndata['color'] = np.array([recolor_map[color]
                                 for color in G_colors]) + current_max_color
    H.ndata['color'] = np.array([recolor_map[color]
                                 for color in H_colors]) + current_max_color
    current_max_color += len(unique_colors)

  return True


def subtree_kernel(G, H, max_iter=10):
  """Calculate the Weisfeiler Lehman graph kernel.

  Arguments:
      G {networkx.classes.graph.Graph} -- Graph
      H {networkx.classes.graph.Graph} -- Graph

  Keyword Arguments:
      max_iter {int} -- The max number of iterations (default: {10})

  Returns:
      float -- The value of the subtree kernel
  """
  # G = dgl.DGLGraph(G)
  # H = dgl.DGLGraph(H)
  # G = dgl.graph(G)
  # H = dgl.graph(H)
  G = dgl.from_networkx(G)
  H = dgl.from_networkx(H)
  kernel_value = 0

  # Set initial colors
  G.ndata['color'] = torch.zeros(G.number_of_nodes())
  H.ndata['color'] = torch.zeros(H.number_of_nodes())
  N = G.number_of_dst_nodes()
  current_max_color = 0

  # G.register_message_func(_send_color)
  # G.register_reduce_func(_gen_create_multiset(N))
  # H.register_message_func(_send_color)
  # H.register_reduce_func(_gen_create_multiset(N))
  G.update_all(_send_color, _gen_create_multiset(N))
  H.update_all(_send_color, _gen_create_multiset(N))

  # Refine colors until convergence
  for i in range(max_iter):
    G_colors = _update_colors(G)
    H_colors = _update_colors(H)

    G_unique_colors, G_counts = np.unique(G_colors, return_counts=True)
    H_unique_colors, H_counts = np.unique(H_colors, return_counts=True)
    G_multiset = dict(zip(G_unique_colors, G_counts))
    H_multiset = dict(zip(H_unique_colors, H_counts))

    # Recoloring (str -> int)
    unique_colors = np.unique(np.append(G_unique_colors, H_unique_colors))
    recolor_map = {color: i + 1 for i, color in enumerate(unique_colors)}

    # Add the value of the subtree kernel in i-th step
    G_color_vector = np.zeros(len(unique_colors))
    H_color_vector = np.zeros(len(unique_colors))
    for color, i in recolor_map.items():
      G_color_vector[i - 1] = G_multiset[color] if color in G_multiset else 0
      H_color_vector[i - 1] = H_multiset[color] if color in H_multiset else 0

    kernel_value += np.dot(G_color_vector, H_color_vector)

    temp = np.array([recolor_map[color] for color in G_colors]) + current_max_color
    G.ndata['color'] = torch.from_numpy(np.array([recolor_map[color] for color in G_colors]) + current_max_color)
    H.ndata['color'] = torch.from_numpy(np.array([recolor_map[color] for color in H_colors]) + current_max_color)
    current_max_color += len(unique_colors)

  return kernel_value
