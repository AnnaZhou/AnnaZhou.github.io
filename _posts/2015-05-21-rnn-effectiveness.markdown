---
layout: post
comments: true
title:  "NOTES ON C3D"
excerpt: "We'll generate C3D features on our own datasets"
date:   2015-12-09 15:00:00
mathjax: true
---


> We'll train RNNs to generate text character by character and ponder the question "how is that even possible?"


## Recurrent Neural Networks

**Sequences**. Depending on your background you might be wondering: *What makes Recurrent Networks so special*? A glaring limitation of Vanilla Neural Networks (and also Convolutional Networks) is that their API is too constrained: they accept a fixed-sized vector as input (e.g. an image) and produce a fixed-sized vector as output (e.g. probabilities of different classes). Not only that: These models perform this mapping using a fixed amount of computational steps (e.g. the number of layers in the model). The core reason that recurrent nets are more exciting is that they allow us to operate over *sequences* of vectors: Sequences in the input, the output, or in the most general case both. A few examples may make this more concrete:




*[[Anti-autism]]

===[[Religion|Religion]]===
*[[French Writings]]
*[[Maria]]
*[[Revelation]]
*[[Mount Agamul]]

== External links==
* [http://www.biblegateway.nih.gov/entrepre/ Website of the World Festival. The labour of India-county defeats at the Ripper of California Road.]

==External links==
* [http://www.romanology.com/ Constitution of the Netherlands and Hispanic Competition for Bilabial and Commonwealth Industry (Republican Constitution of the Extent of the Netherlands)]

{{African American_and_Australian_Parliament{|}}
```

Sometimes the model snaps into a mode of generating random but valid XML:

```
<page>
  <title>Antichrist</title>
  <id>865</id>
  <revision>
    <id>15900676</id>
    <timestamp>2002-08-03T18:14:12Z</timestamp>
    <contributor>
      <username>Paris</username>
      <id>23</id>
    </contributor>
    <minor />
    <comment>Automated conversion</comment>
    <text xml:space="preserve">#REDIRECT [[Christianity]]</text>
  </revision>
</page>
```

The model completely makes up the timestamp, id, and so on. Also, note that it closes the correct tags appropriately and in the correct nested order. Here are [100,000 characters of sampled wikipedia](http://cs.stanford.edu/people/karpathy/char-rnn/wiki.txt) if you're interested to see more.

### Algebraic Geometry (Latex)

The results above suggest that the model is actually quite good at learning complex syntactic structures. Impressed by these results, my labmate ([Justin Johnson](http://cs.stanford.edu/people/jcjohns/)) and I decided to push even further into structured territories and got a hold of [this book](http://stacks.math.columbia.edu/) on algebraic stacks/geometry. We downloaded the raw Latex source file (a 16MB file) and trained a multilayer LSTM. Amazingly, the resulting sampled Latex *almost* compiles. We had to step in and fix a few issues manually but then you get plausible looking math, it's quite astonishing:

<div class="imgcap">
<img src="/assets/rnn/latex4.jpeg" style="border:none;">
<div class="thecap">Sampled (fake) algebraic geometry. <a href="http://cs.stanford.edu/people/jcjohns/fake-math/4.pdf">Here's the actual pdf.</a></div>
</div>

Here's another sample:

<div class="imgcap">
<img src="/assets/rnn/latex3.jpeg" style="border:none;">
<div class="thecap">More hallucinated algebraic geometry. Nice try on the diagram (right).</div>
</div>

As you can see above, sometimes the model tries to generate latex diagrams, but clearly it hasn't really figured them out. I also like the part where it chooses to skip a proof (*"Proof omitted."*, top left). Of course, keep in mind that latex has a relatively difficult structured syntactic format that I haven't even fully mastered myself. For instance, here is a raw sample from the model (unedited):

```
\begin{proof}
We may assume that $\mathcal{I}$ is an abelian sheaf on $\mathcal{C}$.
\item Given a morphism $\Delta : \mathcal{F} \to \mathcal{I}$
is an injective and let $\mathfrak q$ be an abelian sheaf on $X$.
Let $\mathcal{F}$ be a fibered complex. Let $\mathcal{F}$ be a category.
\begin{enumerate}
\item \hyperref[setain-construction-phantom]{Lemma}
\label{lemma-characterize-quasi-finite}
Let $\mathcal{F}$ be an abelian quasi-coherent sheaf on $\mathcal{C}$.
Let $\mathcal{F}$ be a coherent $\mathcal{O}_X$-module. Then
$\mathcal{F}$ is an abelian catenary over $\mathcal{C}$.
\item The following are equivalent
\begin{enumerate}
\item $\mathcal{F}$ is an $\mathcal{O}_X$-module.
\end{lemma}
```

This sample from a relatively decent model illustrates a few common mistakes. For example, the model opens a `\begin{proof}` environment but then ends it with a `\end{lemma}`. This is an example of a problem we'd have to fix manually, and is likely due to the fact that the dependency is too long-term: By the time the model is done with the proof it has forgotten whether it was doing a proof or a lemma. Similarly, it opens an `\begin{enumerate}` but then forgets to close it. We observed that these became less common with larger/better models, but nonetheless, these are the kinds of mistakes that come up.

### Linux Source Code

I wanted to push structured data to its limit, so for the final challenge I decided to use code. In particular, I took all the source and header files found in the [Linux repo on Github](https://github.com/torvalds/linux), concatenated all of them in a single giant file (474MB of C code) (I was originally going to train only on the kernel but that by itself is only ~16MB). Then I trained several as-large-as-fits-on-my-GPU 3-layer LSTMs over a period of a few days. These models have about 10 million parameters, which is still on the lower end for RNN models. The results are superfun:

```c
/*
 * Increment the size file of the new incorrect UI_FILTER group information
 * of the size generatively.
 */
static int indicate_policy(void)
{
  int error;
  if (fd == MARN_EPT) {
    /*
     * The kernel blank will coeld it to userspace.
     */
    if (ss->segment < mem_total)
      unblock_graph_and_set_blocked();
    else
      ret = 1;
    goto bail;
  }
  segaddr = in_SB(in.addr);
  selector = seg / 16;
  setup_works = true;
  for (i = 0; i < blocks; i++) {
    seq = buf[i++];
    bpf = bd->bd.next + i * search;
    if (fd) {
      current = blocked;
    }
  }
  rw->name = "Getjbbregs";
  bprm_self_clearl(&iv->version);
  regs->new = blocks[(BPF_STATS << info->historidac)] | PFMR_CLOBATHINC_SECONDS << 12;
  return segtable;
}
```

The code looks really quite great overall. Of course, I don't think it compiles but when you scroll through the generate code it feels very much like a giant C code base. Notice that the RNN peppers its code with comments here and there at random. It is also very good at making very few syntactic errors. For example, it uses strings properly, pointer notation, etc. It also opens and closes brackets `{[` correctly and learns to indent its code very well. A common error is that it can't keep track of variable names: It often uses undefined variables (e.g. `rw` above), declares variables it never uses (e.g. `int error`), or returns non-existing variables. Lets see a few more examples. Here's another snippet that shows a wider array of operations that the RNN learns:

```c
/*
 * If this error is set, we will need anything right after that BSD.
 */
static void action_new_function(struct s_stat_info *wb)
{
  unsigned long flags;
  int lel_idx_bit = e->edd, *sys & ~((unsigned long) *FIRST_COMPAT);
  buf[0] = 0xFFFFFFFF & (bit << 4);
  min(inc, slist->bytes);
  printk(KERN_WARNING "Memory allocated %02x/%02x, "
    "original MLL instead\n"),
    min(min(multi_run - s->len, max) * num_data_in),
    frame_pos, sz + first_seg);
  div_u64_w(val, inb_p);
  spin_unlock(&disk->queue_lock);
  mutex_unlock(&s->sock->mutex);
  mutex_unlock(&func->mutex);
  return disassemble(info->pending_bh);
}

static void num_serial_settings(struct tty_struct *tty)
{
  if (tty == tty)
    disable_single_st_p(dev);
  pci_disable_spool(port);
  return 0;
}

static void do_command(struct seq_file *m, void *v)
{
  int column = 32 << (cmd[2] & 0x80);
  if (state)
    cmd = (int)(int_state ^ (in_8(&ch->ch_flags) & Cmd) ? 2 : 1);
  else
    seq = 1;
  for (i = 0; i < 16; i++) {
    if (k & (1 << 1))
      pipe = (in_use & UMXTHREAD_UNCCA) +
        ((count & 0x00000000fffffff8) & 0x000000f) << 8;
    if (count == 0)
      sub(pid, ppc_md.kexec_handle, 0x20000000);
    pipe_set_bytes(i, 0);
  }
  /* Free our user pages pointer to place camera if all dash */
  subsystem_info = &of_changes[PAGE_SIZE];
  rek_controls(offset, idx, &soffset);
  /* Now we want to deliberately put it to device */
  control_check_polarity(&context, val, 0);
  for (i = 0; i < COUNTER; i++)
    seq_puts(s, "policy ");
}
```

Notice that in the second function the model compares `tty == tty`, which is vacuously true. On the other hand, at least the variable `tty` exists in the scope this time! In the last function, notice that the code does not return anything, which happens to be correct since the function signature is `void`. However, the first two functions were also declared `void` and did return values. This is again a form of a common mistake due to long-term interactions.

Sometimes the model decides that it's time to sample a new file. This is usually a very amusing part: The model first recites the GNU license character by character, samples a few includes, generates some macros and then dives into the code:

```c
/*
 *  Copyright (c) 2006-2010, Intel Mobile Communications.  All rights reserved.
 *
 *   This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published by
 * the Free Software Foundation.
 *
 *        This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *
 *  GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software Foundation,
 *  Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

#include <linux/kexec.h>
#include <linux/errno.h>
#include <linux/io.h>
#include <linux/platform_device.h>
#include <linux/multi.h>
#include <linux/ckevent.h>

#include <asm/io.h>
#include <asm/prom.h>
#include <asm/e820.h>
#include <asm/system_info.h>
#include <asm/setew.h>
#include <asm/pgproto.h>

#define REG_PG    vesa_slot_addr_pack
#define PFM_NOCOMP  AFSR(0, load)
#define STACK_DDR(type)     (func)

#define SWAP_ALLOCATE(nr)     (e)
#define emulate_sigs()  arch_get_unaligned_child()
#define access_rw(TST)  asm volatile("movd %%esp, %0, %3" : : "r" (0));   \
  if (__type & DO_READ)

static void stat_PC_SEC __read_mostly offsetof(struct seq_argsqueue, \
          pC>[1]);

static void
os_prefix(unsigned long sys)
{
#ifdef CONFIG_PREEMPT
  PUT_PARAM_RAID(2, sel) = get_state_state();
  set_pid_sum((unsigned long)state, current_state_str(),
           (unsigned long)-1->lr_full; low;
}
```

There are too many fun parts to cover- I could probably write an entire blog post on just this part. I'll cut it short for now, but here is [1MB of sampled Linux code](http://cs.stanford.edu/people/karpathy/char-rnn/linux.txt) for your viewing pleasure.

### Generating Baby Names

Lets try one more for fun. Lets feed the RNN a large text file that contains 8000 baby names listed out, one per line (names obtained from [here](http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/)). We can feed this to the RNN and then generate new names! Here are some example names, only showing the ones that do not occur in the training data (90% don't):

*Rudi
Levette
Berice
Lussa
Hany
Mareanne
Chrestina
Carissy
Marylen
Hammine
Janye
Marlise
Jacacrie
Hendred
Romand
Charienna
Nenotto
Ette
Dorane
Wallen
Marly
Darine
Salina
Elvyn
Ersia
Maralena
Minoria
Ellia
Charmin
Antley
Nerille
Chelon
Walmor
Evena
Jeryly
Stachon
Charisa
Allisa
Anatha
Cathanie
Geetra
Alexie
Jerin
Cassen
Herbett
Cossie
Velen
Daurenge
Robester
Shermond
Terisa
Licia
Roselen
Ferine
Jayn
Lusine
Charyanne
Sales
Sanny
Resa
Wallon
Martine
Merus
Jelen
Candica
Wallin
Tel
Rachene
Tarine
Ozila
Ketia
Shanne
Arnande
Karella
Roselina
Alessia
Chasty
Deland
Berther
Geamar
Jackein
Mellisand
Sagdy
Nenc
Lessie
Rasemy
Guen
Gavi
Milea
Anneda
Margoris
Janin
Rodelin
Zeanna
Elyne
Janah
Ferzina
Susta
Pey
Castina*

You can see many more [here](http://cs.stanford.edu/people/karpathy/namesGenUnique.txt). Some of my favorites include "Baby" (haha), "Killie", "Char", "R", "More", "Mars", "Hi", "Saddie", "With" and "Ahbort". Well that was fun.﻿ Of course, you can imagine this being quite useful inspiration when writing a novel, or naming a new startup :)

## Understanding what's going on

We saw that the results at the end of training can be impressive, but how does any of this work? Lets run two quick experiments to briefly peek under the hood.

### The evolution of samples while training

First, it's fun to look at how the sampled text evolves while the model trains. For example, I trained an LSTM of Leo Tolstoy's War and Peace and then generated samples every 100 iterations of training. At iteration 100 the model samples random jumbles:

```
tyntd-iafhatawiaoihrdemot  lytdws  e ,tfti, astai f ogoh eoase rrranbyne 'nhthnee e 
plia tklrgd t o idoe ns,smtt   h ne etie h,hregtrs nigtike,aoaenns lng
```

However, notice that at least it is starting to get an idea about words separated by spaces. Except sometimes it inserts two spaces. It also doesn't know that comma is amost always followed by a space. At 300 iterations we see that the model starts to get an idea about quotes and periods:

```
"Tmont thithey" fomesscerliund
Keushey. Thom here
sheulke, anmerenith ol sivh I lalterthend Bleipile shuwy fil on aseterlome
coaniogennc Phe lism thond hon at. MeiDimorotion in ther thize."
```

The words are now also separated with spaces and the model starts to get the idea about periods at the end of a sentence. At iteration 500:

```
we counter. He stutn co des. His stanted out one ofler that concossions and was 
to gearang reay Jotrets and with fre colt otf paitt thin wall. Which das stimn 
```

the model has now learned to spell the shortest and most common words such as "we", "He", "His", "Which", "and", etc. At iteration 700 we're starting to see more and more English-like text emerge:

```
Aftair fall unsuch that the hall for Prince Velzonski's that me of
her hearly, and behs to so arwage fiving were to it beloge, pavu say falling misfort 
how, and Gogition is so overelical and ofter.
```

At iteration 1200 we're now seeing use of quotations and question/exclamation marks. Longer words have now been learned as well:

```
"Kite vouch!" he repeated by her
door. "But I would be done and quarts, feeling, then, son is people...."
```

Until at last we start to get properly spelled words, quotations, names, and so on by about iteration 2000:

```
"Why do what that day," replied Natasha, and wishing to himself the fact the
princess, Princess Mary was easier, fed in had oftened him.
Pierre aking his soul came to the packs and drove up his father-in-law women.
```

The picture that emerges is that the model first discovers the general word-space structure and then rapidly starts to learn the words; First starting with the short words and then eventually the longer ones. Topics and themes that span multiple words (and in general longer-term dependencies) start to emerge only much later.

### Visualizing the predictions and the "neuron" firings in the RNN

Another fun visualization is to look at the predicted distributions over characters. In the visualizations below we feed a Wikipedia RNN model character data from the validation set (shown along the blue/green rows) and under every character we visualize (in red) the top 5 guesses that the model assigns for the next character. The guesses are colored by their probability (so dark red = judged as very likely, white = not very likely). For example, notice that there are stretches of characters where the model is extremely confident about the next letter (e.g., the model is very confident about characters during the *http://www.* sequence).

The input character sequence (blue/green) is colored based on the *firing* of a randomly chosen neuron in the hidden representation of the RNN. Think about it as green = very excited and blue = not very excited (for those familiar with details of LSTMs, these are values between [-1,1] in the hidden state vector, which is just the gated and tanh'd LSTM cell state). Intuitively, this is visualizing the firing rate of some neuron in the "brain" of the RNN while it reads the input sequence. Different neurons might be looking for different patterns; Below we'll look at 4 different ones that I found and thought were interesting or interpretable (many also aren't):

<div class="imgcap">
<img src="/assets/rnn/under1.jpeg" style="border:none;">
<div class="thecap">
The neuron highlighted in this image seems to get very excited about URLs and turns off outside of the URLs. The LSTM is likely using this neuron to remember if it is inside a URL or not.
</div>
</div>

<div class="imgcap">
<img src="/assets/rnn/under2.jpeg" style="border:none;">
<div class="thecap">
The highlighted neuron here gets very excited when the RNN is inside the [[ ]] markdown environment and turns off outside of it. Interestingly, the neuron can't turn on right after it sees the character "[", it must wait for the second "[" and then activate. This task of counting whether the model has seen one or two "[" is likely done with a different neuron.
</div>
</div>

<div class="imgcap">
<img src="/assets/rnn/under3.jpeg" style="border:none;">
<div class="thecap">
Here we see a neuron that varies seemingly linearly across the [[ ]] environment. In other words its activation is giving the RNN a time-aligned coordinate system across the [[ ]] scope. The RNN can use this information to make different characters more or less likely depending on how early/late it is in the [[ ]] scope (perhaps?).
</div>
</div>

<div class="imgcap">
<img src="/assets/rnn/under4.jpeg" style="border:none;">
<div class="thecap">
Here is another neuron that has very local behavior: it is relatively silent but sharply turns off right after the first "w" in the "www" sequence. The RNN might be using this neuron to count up how far in the "www" sequence it is, so that it can know whether it should emit another "w", or if it should start the URL.
</div>
</div>

Of course, a lot of these conclusions are slightly hand-wavy as the hidden state of the RNN is a huge, high-dimensional and largely distributed representation. These visualizations were produced with custom HTML/CSS/Javascript, you can see a sketch of what's involved [here](http://cs.stanford.edu/people/karpathy/viscode.zip) if you'd like to create something similar.

We can also condense this visualization by excluding the most likely predictions and only visualize the text, colored by activations of a cell. We can see that in addition to a large portion of cells that do not do anything interpretible, about 5% of them turn out to have learned quite interesting and interpretible algorithms:

<div class="imgcap">
<img src="/assets/rnn/pane1.png" style="border:none;max-width:100%">
<img src="/assets/rnn/pane2.png" style="border:none;max-width:100%">
<div class="thecap">
</div>
</div>

Again, what is beautiful about this is that we didn't have to hardcode at any point that if you're trying to predict the next character it might, for example, be useful to keep track of whether or not you are currently inside or outside of quote. We just trained the LSTM on raw data and it decided that this is a useful quantitity to keep track of. In other words one of its cells gradually tuned itself during training to become a quote detection cell, since this helps it better perform the final task. This is one of the cleanest and most compelling examples of where the power in Deep Learning models (and more generally end-to-end training) is coming from.

## Source Code

I hope I've convinced you that training character-level language models is a very fun exercise. You can train your own models using the [char-rnn code](https://github.com/karpathy/char-rnn) I released on Github (under MIT license). It takes one large text file and trains a character-level model that you can then sample from. Also, it helps if you have a GPU or otherwise training on CPU will be about a factor of 10x slower. In any case, if you end up training on some data and getting fun results let me know! And if you get lost in the Torch/Lua codebase remember that all it is is just a more fancy version of this [100-line gist](https://gist.github.com/karpathy/d4dee566867f8291f086).

*Brief digression.* The code is written in [Torch 7](http://torch.ch/), which has recently become my favorite deep learning framework. I've only started working with Torch/LUA over the last few months and it hasn't been easy (I spent a good amount of time digging through the raw Torch code on Github and asking questions on their *gitter* to get things done), but once you get a hang of things it offers a lot of flexibility and speed. I've also worked with Caffe and Theano in the past and I believe Torch, while not perfect, gets its levels of abstraction and philosophy right better than others. In my view the desirable features of an effective framework are: 

1. CPU/GPU transparent Tensor library with a lot of functionality (slicing, array/matrix operations, etc. )
2. An entirely separate code base in a scripting language (ideally Python) that operates over Tensors and implements all Deep Learning stuff (forward/backward, computation graphs, etc)
3. It should be possible to easily share pretrained models (Caffe does this well, others don't), and crucially 
4. NO compilation step (or at least not as currently done in Theano). The trend in Deep Learning is towards larger, more complex networks that are are time-unrolled in complex graphs. It is critical that these do not compile for a long time or development time greatly suffers. Second, by compiling one gives up interpretability and the ability to log/debug effectively. If there is an *option* to compile the graph once it has been developed for efficiency in prod that's fine.

## Further Reading

Before the end of the post I also wanted to position RNNs in a wider context and provide a sketch of the current research directions. RNNs have recently generated a significant amount of buzz and excitement in the field of Deep Learning. Similar to Convolutional Networks they have been around for decades but their full potential has only recently started to get widely recognized, in large part due to our growing computational resources. Here's a brief sketch of a few recent developments (definitely not complete list, and a lot of this work draws from research back to 1990s, see related work sections):

In the domain of **NLP/Speech**, RNNs [transcribe speech to text](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf), perform [machine translation](http://arxiv.org/abs/1409.3215), [generate handwritten text](http://www.cs.toronto.edu/~graves/handwriting.html), and of course, they have been used as powerful language models [(Sutskever et al.)](http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf) [(Graves)](http://arxiv.org/abs/1308.0850) [(Mikolov et al.)](http://www.rnnlm.org/) (both on the level of characters and words). Currently it seems that word-level models work better than character-level models, but this is surely a temporary thing.

**Computer Vision.** RNNs are also quickly becoming pervasive in Computer Vision. For example, we're seeing RNNs in frame-level [video classification](http://arxiv.org/abs/1411.4389), [image captioning](http://arxiv.org/abs/1411.4555) (also including my own work and many others), [video captioning](http://arxiv.org/abs/1505.00487) and very recently [visual question answering](http://arxiv.org/abs/1505.02074). My personal favorite RNNs in Computer Vision paper is [Recurrent Models of Visual Attention](http://arxiv.org/abs/1406.6247), both due to its high-level direction (sequential processing of images with glances) and the low-level modeling (REINFORCE learning rule that is a special case of policy gradient methods in Reinforcement Learning, which allows one to train models that perform non-differentiable computation (taking glances around the image in this case)). I'm confident that this type of hybrid model that consists of a blend of CNN for raw perception coupled with an RNN glance policy on top will become pervasive in perception, especially for more complex tasks that go beyond classifying some objects in plain view.

**Inductive Reasoning, Memories and Attention.** Another extremely exciting direction of research is oriented towards addressing the limitations of vanilla recurrent networks. One problem is that RNNs are not inductive: They memorize sequences extremely well, but they don't necessarily always show convincing signs of generalizing in the *correct* way (I'll provide pointers in a bit that make this more concrete). A second issue is they unnecessarily couple their representation size to the amount of computation per step. For instance, if you double the size of the hidden state vector you'd quadruple the amount of FLOPS at each step due to the matrix multiplication. Ideally, we'd like to maintain a huge representation/memory (e.g. containing all of Wikipedia or many intermediate state variables), while maintaining the ability to keep computation per time step fixed.

The first convincing example of moving towards these directions was developed in DeepMind's [Neural Turing Machines](http://arxiv.org/abs/1410.5401) paper. This paper sketched a path towards models that can perform read/write operations between large, external memory arrays and a smaller set of memory registers (think of these as our working memory) where the computation happens. Crucially, the NTM paper also featured very interesting memory addressing mechanisms that were implemented with a (soft, and fully-differentiable) attention model. The concept of **soft attention** has turned out to be a powerful modeling feature and was also featured in [Neural Machine Translation by Jointly Learning to Align and Translate](http://arxiv.org/abs/1409.0473) for Machine Translation and [Memory Networks](http://arxiv.org/abs/1503.08895) for (toy) Question Answering. In fact, I'd go as far as to say that 

> The concept of **attention** is the most interesting recent architectural innovation in neural networks.

Now, I don't want to dive into too many details but a soft attention scheme for memory addressing is convenient because it keeps the model fully-differentiable, but unfortunately one sacrifices efficiency because everything that can be attended to is attended to (but softly). Think of this as declaring a pointer in C that doesn't point to a specific address but instead defines an entire distribution over all addresses in the entire memory, and dereferencing the pointer returns a weighted sum of the pointed content (that would be an expensive operation!). This has motivated multiple authors to swap soft attention models for **hard attention** where one samples a particular chunk of memory to attend to (e.g. a read/write action for some memory cell instead of reading/writing from all cells to some degree). This model is significantly more philosophically appealing, scalable and efficient, but unfortunately it is also non-differentiable. This then calls for use of techniques from the Reinforcement Learning literature (e.g. REINFORCE) where people are perfectly used to the concept of non-differentiable interactions. This is very much ongoing work but these hard attention models have been explored, for example, in [Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](http://arxiv.org/abs/1503.01007), [Reinforcement Learning Neural Turing Machines](http://arxiv.org/abs/1505.00521), and [Show Attend and Tell](http://arxiv.org/abs/1502.03044).

**People**. If you'd like to read up on RNNs I recommend theses from [Alex Graves](http://www.cs.toronto.edu/~graves/), [Ilya Sutskever](http://www.cs.toronto.edu/~ilya/) and [Tomas Mikolov](http://www.rnnlm.org/). For more about REINFORCE and more generally Reinforcement Learning and policy gradient methods (which REINFORCE is a special case of) [David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Home.html)'s class, or one of [Pieter Abbeel](http://www.cs.berkeley.edu/~pabbeel/)'s classes.

**Code**. If you'd like to play with training RNNs I hear good things about [keras](https://github.com/fchollet/keras) or [passage](https://github.com/IndicoDataSolutions/Passage) for Theano, the [code](https://github.com/karpathy/char-rnn) released with this post for Torch, or [this gist](https://gist.github.com/karpathy/587454dc0146a6ae21fc) for raw numpy code I wrote a while ago that implements an efficient, batched LSTM forward and backward pass. You can also have a look at my numpy-based [NeuralTalk](https://github.com/karpathy/neuraltalk) which uses an RNN/LSTM to caption images, or maybe this [Caffe](http://jeffdonahue.com/lrcn/) implementation by Jeff Donahue.

## Conclusion

We've learned about RNNs, how they work, why they have become a big deal, we've trained an RNN character-level language model on several fun datasets, and we've seen where RNNs are going. You can confidently expect a large amount of innovation in the space of RNNs, and I believe they will become a pervasive and critical component to intelligent systems.

Lastly, to add some **meta** to this post, I trained an RNN on the source file of this blog post. Unfortunately, at about 46K characters I haven't written enough data to properly feed the RNN, but the returned sample (generated with low temperature to get a more typical sample) is:

```
I've the RNN with and works, but the computed with program of the 
RNN with and the computed of the RNN with with and the code
```

Yes, the post was about RNN and how well it works, so clearly this works :). See you next time!

**EDIT (extra links):** 

Videos:

- I gave a talk on this work at the [London Deep Learning meetup (video)](https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks).

Discussions:

- [HN discussion](https://news.ycombinator.com/item?id=9584325)
- Reddit discussion on [r/machinelearning](http://www.reddit.com/r/MachineLearning/comments/36s673/the_unreasonable_effectiveness_of_recurrent/)
- Reddit discussion on [r/programming](http://www.reddit.com/r/programming/comments/36su8d/the_unreasonable_effectiveness_of_recurrent/)

Replies:

- [Yoav Goldberg](https://twitter.com/yoavgo) compared these RNN results to [n-gram maximum likelihood (counting) baseline](http://nbviewer.ipython.org/gist/yoavg/d76121dfde2618422139)
- [@nylk](https://twitter.com/nylk) trained char-rnn on [cooking recipes](https://gist.github.com/nylki/1efbaa36635956d35bcc). They look great!
- [@MrChrisJohnson](https://twitter.com/MrChrisJohnson) trained char-rnn on Eminem lyrics and then synthesized a rap song with robotic voice reading it out. Hilarious :)
- [@samim](https://twitter.com/samim) trained char-rnn on [Obama Speeches](https://medium.com/@samim/obama-rnn-machine-generated-political-speeches-c8abd18a2ea0). They look fun!
- [João Felipe](https://twitter.com/seaandsailor) trained char-rnn irish folk music and [sampled music](https://soundcloud.com/seaandsailor/sets/char-rnn-composes-irish-folk-music)
- [Bob Sturm](https://twitter.com/boblsturm) also trained char-rnn on [music in ABC notation](https://highnoongmt.wordpress.com/2015/05/22/lisls-stis-recurrent-neural-networks-for-folk-music-generation/)
- [RNN Bible bot](https://twitter.com/RNN_Bible) by [Maximilien](https://twitter.com/the__glu/with_replies)
- [Learning Holiness](http://cpury.github.io/learning-holiness/) learning the Bible
- [Terminal.com snapshot](https://www.terminal.com/tiny/ZMcqdkWGOM) that has char-rnn set up and ready to go in a browser-based virtual machine (thanks [@samim](https://www.twitter.com/samim))

