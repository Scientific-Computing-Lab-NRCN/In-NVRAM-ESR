# Makefile.in generated by automake 1.16.4 from Makefile.am.
# examples/Makefile.  Generated from Makefile.in by configure.

# Copyright (C) 1994-2021 Free Software Foundation, Inc.

# This Makefile.in is free software; the Free Software Foundation
# gives unlimited permission to copy and/or distribute it,
# with or without modifications, as long as this notice is preserved.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, to the extent permitted by law; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.



#
# Copyright 2014-2016, Gdansk University of Technology
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holder nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


am__is_gnu_make = { \
  if test -z '$(MAKELEVEL)'; then \
    false; \
  elif test -n '$(MAKE_HOST)'; then \
    true; \
  elif test -n '$(MAKE_VERSION)' && test -n '$(CURDIR)'; then \
    true; \
  else \
    false; \
  fi; \
}
am__make_running_with_option = \
  case $${target_option-} in \
      ?) ;; \
      *) echo "am__make_running_with_option: internal error: invalid" \
              "target option '$${target_option-}' specified" >&2; \
         exit 1;; \
  esac; \
  has_opt=no; \
  sane_makeflags=$$MAKEFLAGS; \
  if $(am__is_gnu_make); then \
    sane_makeflags=$$MFLAGS; \
  else \
    case $$MAKEFLAGS in \
      *\\[\ \	]*) \
        bs=\\; \
        sane_makeflags=`printf '%s\n' "$$MAKEFLAGS" \
          | sed "s/$$bs$$bs[$$bs $$bs	]*//g"`;; \
    esac; \
  fi; \
  skip_next=no; \
  strip_trailopt () \
  { \
    flg=`printf '%s\n' "$$flg" | sed "s/$$1.*$$//"`; \
  }; \
  for flg in $$sane_makeflags; do \
    test $$skip_next = yes && { skip_next=no; continue; }; \
    case $$flg in \
      *=*|--*) continue;; \
        -*I) strip_trailopt 'I'; skip_next=yes;; \
      -*I?*) strip_trailopt 'I';; \
        -*O) strip_trailopt 'O'; skip_next=yes;; \
      -*O?*) strip_trailopt 'O';; \
        -*l) strip_trailopt 'l'; skip_next=yes;; \
      -*l?*) strip_trailopt 'l';; \
      -[dEDm]) skip_next=yes;; \
      -[JT]) skip_next=yes;; \
    esac; \
    case $$flg in \
      *$$target_option*) has_opt=yes; break;; \
    esac; \
  done; \
  test $$has_opt = yes
am__make_dryrun = (target_option=n; $(am__make_running_with_option))
am__make_keepgoing = (target_option=k; $(am__make_running_with_option))
pkgdatadir = $(datadir)/mpi_pmem
pkgincludedir = $(includedir)/mpi_pmem
pkglibdir = $(libdir)/mpi_pmem
pkglibexecdir = $(libexecdir)/mpi_pmem
am__cd = CDPATH="$${ZSH_VERSION+.}$(PATH_SEPARATOR)" && cd
install_sh_DATA = $(install_sh) -c -m 644
install_sh_PROGRAM = $(install_sh) -c
install_sh_SCRIPT = $(install_sh) -c
INSTALL_HEADER = $(INSTALL_DATA)
transform = $(program_transform_name)
NORMAL_INSTALL = :
PRE_INSTALL = :
POST_INSTALL = :
NORMAL_UNINSTALL = :
PRE_UNINSTALL = :
POST_UNINSTALL = :
build_triplet = x86_64-unknown-linux-gnu
host_triplet = x86_64-unknown-linux-gnu
bin_PROGRAMS = mpi_one_sided$(EXEEXT) HPCCG/HPCCG$(EXEEXT) \
	HPCCG/HPCCG_checkpoint_disk$(EXEEXT) \
	HPCCG/HPCCG_checkpoint_pmem$(EXEEXT) \
	HPCCG/HPCCG_checkpoint_double_buffered$(EXEEXT)
subdir = examples
ACLOCAL_M4 = $(top_srcdir)/aclocal.m4
am__aclocal_m4_deps = $(top_srcdir)/m4/ax_prog_cc_mpi.m4 \
	$(top_srcdir)/m4/ax_prog_cxx_mpi.m4 \
	$(top_srcdir)/m4/libtool.m4 $(top_srcdir)/m4/ltoptions.m4 \
	$(top_srcdir)/m4/ltsugar.m4 $(top_srcdir)/m4/ltversion.m4 \
	$(top_srcdir)/m4/lt~obsolete.m4 $(top_srcdir)/configure.ac
am__configure_deps = $(am__aclocal_m4_deps) $(CONFIGURE_DEPENDENCIES) \
	$(ACLOCAL_M4)
DIST_COMMON = $(srcdir)/Makefile.am $(am__DIST_COMMON)
mkinstalldirs = $(install_sh) -d
CONFIG_CLEAN_FILES =
CONFIG_CLEAN_VPATH_FILES =
am__installdirs = "$(DESTDIR)$(bindir)"
PROGRAMS = $(bin_PROGRAMS)
am__dirstamp = $(am__leading_dot)dirstamp
am_HPCCG_HPCCG_OBJECTS = HPCCG/compute_residual.$(OBJEXT) \
	HPCCG/ddot.$(OBJEXT) HPCCG/dump_matlab_matrix.$(OBJEXT) \
	HPCCG/exchange_externals.$(OBJEXT) \
	HPCCG/generate_matrix.$(OBJEXT) \
	HPCCG/HPC_Sparse_Matrix.$(OBJEXT) HPCCG/HPC_sparsemv.$(OBJEXT) \
	HPCCG/HPCCG.$(OBJEXT) HPCCG/make_local_matrix.$(OBJEXT) \
	HPCCG/mytimer.$(OBJEXT) HPCCG/read_HPC_row.$(OBJEXT) \
	HPCCG/waxpby.$(OBJEXT) HPCCG/YAML_Doc.$(OBJEXT) \
	HPCCG/YAML_Element.$(OBJEXT) HPCCG/HPCCG_main.$(OBJEXT)
HPCCG_HPCCG_OBJECTS = $(am_HPCCG_HPCCG_OBJECTS)
HPCCG_HPCCG_LDADD = $(LDADD)
HPCCG_HPCCG_DEPENDENCIES =  \
	../src/mpi_one_sided_extension/libmpi-pmem-one-sided.la
AM_V_lt = $(am__v_lt_$(V))
am__v_lt_ = $(am__v_lt_$(AM_DEFAULT_VERBOSITY))
am__v_lt_0 = --silent
am__v_lt_1 = 
am_HPCCG_HPCCG_checkpoint_disk_OBJECTS =  \
	HPCCG/compute_residual.$(OBJEXT) HPCCG/ddot.$(OBJEXT) \
	HPCCG/dump_matlab_matrix.$(OBJEXT) \
	HPCCG/exchange_externals.$(OBJEXT) \
	HPCCG/generate_matrix.$(OBJEXT) \
	HPCCG/HPC_Sparse_Matrix.$(OBJEXT) HPCCG/HPC_sparsemv.$(OBJEXT) \
	HPCCG/HPCCG_checkpoint_disk.$(OBJEXT) \
	HPCCG/make_local_matrix.$(OBJEXT) HPCCG/mytimer.$(OBJEXT) \
	HPCCG/read_HPC_row.$(OBJEXT) HPCCG/waxpby.$(OBJEXT) \
	HPCCG/YAML_Doc.$(OBJEXT) HPCCG/YAML_Element.$(OBJEXT) \
	HPCCG/HPCCG_main.$(OBJEXT)
HPCCG_HPCCG_checkpoint_disk_OBJECTS =  \
	$(am_HPCCG_HPCCG_checkpoint_disk_OBJECTS)
HPCCG_HPCCG_checkpoint_disk_LDADD = $(LDADD)
HPCCG_HPCCG_checkpoint_disk_DEPENDENCIES =  \
	../src/mpi_one_sided_extension/libmpi-pmem-one-sided.la
am_HPCCG_HPCCG_checkpoint_double_buffered_OBJECTS =  \
	HPCCG/compute_residual.$(OBJEXT) HPCCG/ddot.$(OBJEXT) \
	HPCCG/dump_matlab_matrix.$(OBJEXT) \
	HPCCG/exchange_externals.$(OBJEXT) \
	HPCCG/generate_matrix.$(OBJEXT) \
	HPCCG/HPC_Sparse_Matrix.$(OBJEXT) HPCCG/HPC_sparsemv.$(OBJEXT) \
	HPCCG/HPCCG_checkpoint_double_buffered.$(OBJEXT) \
	HPCCG/make_local_matrix.$(OBJEXT) HPCCG/mytimer.$(OBJEXT) \
	HPCCG/read_HPC_row.$(OBJEXT) HPCCG/waxpby.$(OBJEXT) \
	HPCCG/YAML_Doc.$(OBJEXT) HPCCG/YAML_Element.$(OBJEXT) \
	HPCCG/HPCCG_main.$(OBJEXT)
HPCCG_HPCCG_checkpoint_double_buffered_OBJECTS =  \
	$(am_HPCCG_HPCCG_checkpoint_double_buffered_OBJECTS)
HPCCG_HPCCG_checkpoint_double_buffered_LDADD = $(LDADD)
HPCCG_HPCCG_checkpoint_double_buffered_DEPENDENCIES =  \
	../src/mpi_one_sided_extension/libmpi-pmem-one-sided.la
am_HPCCG_HPCCG_checkpoint_pmem_OBJECTS =  \
	HPCCG/compute_residual.$(OBJEXT) HPCCG/ddot.$(OBJEXT) \
	HPCCG/dump_matlab_matrix.$(OBJEXT) \
	HPCCG/exchange_externals.$(OBJEXT) \
	HPCCG/generate_matrix.$(OBJEXT) \
	HPCCG/HPC_Sparse_Matrix.$(OBJEXT) HPCCG/HPC_sparsemv.$(OBJEXT) \
	HPCCG/HPCCG_checkpoint_pmem.$(OBJEXT) \
	HPCCG/make_local_matrix.$(OBJEXT) HPCCG/mytimer.$(OBJEXT) \
	HPCCG/read_HPC_row.$(OBJEXT) HPCCG/waxpby.$(OBJEXT) \
	HPCCG/YAML_Doc.$(OBJEXT) HPCCG/YAML_Element.$(OBJEXT) \
	HPCCG/HPCCG_main.$(OBJEXT)
HPCCG_HPCCG_checkpoint_pmem_OBJECTS =  \
	$(am_HPCCG_HPCCG_checkpoint_pmem_OBJECTS)
HPCCG_HPCCG_checkpoint_pmem_LDADD = $(LDADD)
HPCCG_HPCCG_checkpoint_pmem_DEPENDENCIES =  \
	../src/mpi_one_sided_extension/libmpi-pmem-one-sided.la
am_mpi_one_sided_OBJECTS = main.$(OBJEXT)
mpi_one_sided_OBJECTS = $(am_mpi_one_sided_OBJECTS)
mpi_one_sided_LDADD = $(LDADD)
mpi_one_sided_DEPENDENCIES =  \
	../src/mpi_one_sided_extension/libmpi-pmem-one-sided.la
AM_V_P = $(am__v_P_$(V))
am__v_P_ = $(am__v_P_$(AM_DEFAULT_VERBOSITY))
am__v_P_0 = false
am__v_P_1 = :
AM_V_GEN = $(am__v_GEN_$(V))
am__v_GEN_ = $(am__v_GEN_$(AM_DEFAULT_VERBOSITY))
am__v_GEN_0 = @echo "  GEN     " $@;
am__v_GEN_1 = 
AM_V_at = $(am__v_at_$(V))
am__v_at_ = $(am__v_at_$(AM_DEFAULT_VERBOSITY))
am__v_at_0 = @
am__v_at_1 = 
DEFAULT_INCLUDES = -I.
depcomp = $(SHELL) $(top_srcdir)/depcomp
am__maybe_remake_depfiles = depfiles
am__depfiles_remade = ./$(DEPDIR)/main.Po HPCCG/$(DEPDIR)/HPCCG.Po \
	HPCCG/$(DEPDIR)/HPCCG_checkpoint_disk.Po \
	HPCCG/$(DEPDIR)/HPCCG_checkpoint_double_buffered.Po \
	HPCCG/$(DEPDIR)/HPCCG_checkpoint_pmem.Po \
	HPCCG/$(DEPDIR)/HPCCG_main.Po \
	HPCCG/$(DEPDIR)/HPC_Sparse_Matrix.Po \
	HPCCG/$(DEPDIR)/HPC_sparsemv.Po HPCCG/$(DEPDIR)/YAML_Doc.Po \
	HPCCG/$(DEPDIR)/YAML_Element.Po \
	HPCCG/$(DEPDIR)/compute_residual.Po HPCCG/$(DEPDIR)/ddot.Po \
	HPCCG/$(DEPDIR)/dump_matlab_matrix.Po \
	HPCCG/$(DEPDIR)/exchange_externals.Po \
	HPCCG/$(DEPDIR)/generate_matrix.Po \
	HPCCG/$(DEPDIR)/make_local_matrix.Po \
	HPCCG/$(DEPDIR)/mytimer.Po HPCCG/$(DEPDIR)/read_HPC_row.Po \
	HPCCG/$(DEPDIR)/waxpby.Po
am__mv = mv -f
COMPILE = $(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS) $(CFLAGS)
LTCOMPILE = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
	$(LIBTOOLFLAGS) --mode=compile $(CC) $(DEFS) \
	$(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) \
	$(AM_CFLAGS) $(CFLAGS)
AM_V_CC = $(am__v_CC_$(V))
am__v_CC_ = $(am__v_CC_$(AM_DEFAULT_VERBOSITY))
am__v_CC_0 = @echo "  CC      " $@;
am__v_CC_1 = 
CCLD = $(CC)
LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
	$(LIBTOOLFLAGS) --mode=link $(CCLD) $(AM_CFLAGS) $(CFLAGS) \
	$(AM_LDFLAGS) $(LDFLAGS) -o $@
AM_V_CCLD = $(am__v_CCLD_$(V))
am__v_CCLD_ = $(am__v_CCLD_$(AM_DEFAULT_VERBOSITY))
am__v_CCLD_0 = @echo "  CCLD    " $@;
am__v_CCLD_1 = 
CXXCOMPILE = $(CXX) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	$(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS)
LTCXXCOMPILE = $(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) \
	$(LIBTOOLFLAGS) --mode=compile $(CXX) $(DEFS) \
	$(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) \
	$(AM_CXXFLAGS) $(CXXFLAGS)
AM_V_CXX = $(am__v_CXX_$(V))
am__v_CXX_ = $(am__v_CXX_$(AM_DEFAULT_VERBOSITY))
am__v_CXX_0 = @echo "  CXX     " $@;
am__v_CXX_1 = 
CXXLD = $(CXX)
CXXLINK = $(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) \
	$(LIBTOOLFLAGS) --mode=link $(CXXLD) $(AM_CXXFLAGS) \
	$(CXXFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@
AM_V_CXXLD = $(am__v_CXXLD_$(V))
am__v_CXXLD_ = $(am__v_CXXLD_$(AM_DEFAULT_VERBOSITY))
am__v_CXXLD_0 = @echo "  CXXLD   " $@;
am__v_CXXLD_1 = 
SOURCES = $(HPCCG_HPCCG_SOURCES) \
	$(HPCCG_HPCCG_checkpoint_disk_SOURCES) \
	$(HPCCG_HPCCG_checkpoint_double_buffered_SOURCES) \
	$(HPCCG_HPCCG_checkpoint_pmem_SOURCES) \
	$(mpi_one_sided_SOURCES)
DIST_SOURCES = $(HPCCG_HPCCG_SOURCES) \
	$(HPCCG_HPCCG_checkpoint_disk_SOURCES) \
	$(HPCCG_HPCCG_checkpoint_double_buffered_SOURCES) \
	$(HPCCG_HPCCG_checkpoint_pmem_SOURCES) \
	$(mpi_one_sided_SOURCES)
am__can_run_installinfo = \
  case $$AM_UPDATE_INFO_DIR in \
    n|no|NO) false;; \
    *) (install-info --version) >/dev/null 2>&1;; \
  esac
am__tagged_files = $(HEADERS) $(SOURCES) $(TAGS_FILES) $(LISP)
# Read a list of newline-separated strings from the standard input,
# and print each of them once, without duplicates.  Input order is
# *not* preserved.
am__uniquify_input = $(AWK) '\
  BEGIN { nonempty = 0; } \
  { items[$$0] = 1; nonempty = 1; } \
  END { if (nonempty) { for (i in items) print i; }; } \
'
# Make sure the list of sources is unique.  This is necessary because,
# e.g., the same source file might be shared among _SOURCES variables
# for different programs/libraries.
am__define_uniq_tagged_files = \
  list='$(am__tagged_files)'; \
  unique=`for i in $$list; do \
    if test -f "$$i"; then echo $$i; else echo $(srcdir)/$$i; fi; \
  done | $(am__uniquify_input)`
am__DIST_COMMON = $(srcdir)/Makefile.in $(top_srcdir)/depcomp
DISTFILES = $(DIST_COMMON) $(DIST_SOURCES) $(TEXINFOS) $(EXTRA_DIST)
ACLOCAL = ${SHELL} '/home/yonif/A_PAZI/ESR-CG/mpi-pmem-ext-harel/mpi_one_sided_extension/missing' aclocal-1.16
AMTAR = $${TAR-tar}
AM_CFLAGS = -std=c99 -W -Wall -fopenmp
AM_CPPFLAGS = -D_GNU_SOURCE -D_LOG_ERROR -I$(srcdir)/../src -DUSING_OMP
AM_CXXFLAGS = -W -Wall -DUSING_MPI -fopenmp
AM_DEFAULT_VERBOSITY = 1
AM_LDFLAGS = 
AR = ar
AUTOCONF = ${SHELL} '/home/yonif/A_PAZI/ESR-CG/mpi-pmem-ext-harel/mpi_one_sided_extension/missing' autoconf
AUTOHEADER = ${SHELL} '/home/yonif/A_PAZI/ESR-CG/mpi-pmem-ext-harel/mpi_one_sided_extension/missing' autoheader
AUTOMAKE = ${SHELL} '/home/yonif/A_PAZI/ESR-CG/mpi-pmem-ext-harel/mpi_one_sided_extension/missing' automake-1.16
AWK = gawk
CC = /opt/ohpc/pub/mpi/mpich-gnu-ohpc/3.2.1/bin/mpicc
CCDEPMODE = depmode=gcc3
CFLAGS = -g -O2
CPP = /opt/ohpc/pub/mpi/mpich-gnu-ohpc/3.2.1/bin/mpicc -E
CPPFLAGS = 
CSCOPE = cscope
CTAGS = ctags
CXX = /opt/ohpc/pub/mpi/mpich-gnu-ohpc/3.2.1/bin/mpic++
CXXCPP = /opt/ohpc/pub/mpi/mpich-gnu-ohpc/3.2.1/bin/mpic++ -E
CXXDEPMODE = depmode=gcc3
CXXFLAGS = -g -O2
CYGPATH_W = echo
DEFS = -DPACKAGE_NAME=\"mpi_pmem\" -DPACKAGE_TARNAME=\"mpi_pmem\" -DPACKAGE_VERSION=\"1.0\" -DPACKAGE_STRING=\"mpi_pmem\ 1.0\" -DPACKAGE_BUGREPORT=\"\" -DPACKAGE_URL=\"\" -DSTDC_HEADERS=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STRINGS_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_STDINT_H=1 -DHAVE_UNISTD_H=1 -D__EXTENSIONS__=1 -D_ALL_SOURCE=1 -D_GNU_SOURCE=1 -D_POSIX_PTHREAD_SEMANTICS=1 -D_TANDEM_SOURCE=1 -DPACKAGE=\"mpi_pmem\" -DVERSION=\"1.0\" -DHAVE_MPI=1 -DHAVE_MPI=1 -DHAVE_DLFCN_H=1 -DLT_OBJDIR=\".libs/\"
DEPDIR = .deps
DLLTOOL = false
DSYMUTIL = 
DUMPBIN = 
ECHO_C = 
ECHO_N = -n
ECHO_T = 
EGREP = /usr/bin/grep -E
ETAGS = etags
EXEEXT = 
FGREP = /usr/bin/grep -F
GREP = /usr/bin/grep
INSTALL = /usr/bin/install -c
INSTALL_DATA = ${INSTALL} -m 644
INSTALL_PROGRAM = ${INSTALL}
INSTALL_SCRIPT = ${INSTALL}
INSTALL_STRIP_PROGRAM = $(install_sh) -c -s
LD = /usr/bin/ld -m elf_x86_64
LDADD = -lpthread -lpmem \
	../src/mpi_one_sided_extension/libmpi-pmem-one-sided.la
LDFLAGS = 
LIBOBJS = 
LIBS = 
LIBTOOL = $(SHELL) $(top_builddir)/libtool
LIPO = 
LN_S = ln -s
LTLIBOBJS = 
LT_SYS_LIBRARY_PATH = 
MAKEINFO = ${SHELL} '/home/yonif/A_PAZI/ESR-CG/mpi-pmem-ext-harel/mpi_one_sided_extension/missing' makeinfo
MANIFEST_TOOL = :
MKDIR_P = /usr/bin/mkdir -p
MPICC = 
MPICXX = 
NM = /usr/bin/nm -B
NMEDIT = 
OBJDUMP = objdump
OBJEXT = o
OTOOL = 
OTOOL64 = 
PACKAGE = mpi_pmem
PACKAGE_BUGREPORT = 
PACKAGE_NAME = mpi_pmem
PACKAGE_STRING = mpi_pmem 1.0
PACKAGE_TARNAME = mpi_pmem
PACKAGE_URL = 
PACKAGE_VERSION = 1.0
PATH_SEPARATOR = :
RANLIB = ranlib
SED = /usr/bin/sed
SET_MAKE = 
SHELL = /bin/sh
STRIP = strip
VERSION = 1.0
abs_builddir = /home/yonif/A_PAZI/ESR-CG/mpi-pmem-ext-harel/mpi_one_sided_extension/examples
abs_srcdir = /home/yonif/A_PAZI/ESR-CG/mpi-pmem-ext-harel/mpi_one_sided_extension/examples
abs_top_builddir = /home/yonif/A_PAZI/ESR-CG/mpi-pmem-ext-harel/mpi_one_sided_extension
abs_top_srcdir = /home/yonif/A_PAZI/ESR-CG/mpi-pmem-ext-harel/mpi_one_sided_extension
ac_ct_AR = ar
ac_ct_CC = /opt/ohpc/pub/mpi/mpich-gnu-ohpc/3.2.1/bin/mpicc
ac_ct_CXX = 
ac_ct_DUMPBIN = 
am__include = include
am__leading_dot = .
am__quote = 
am__tar = $${TAR-tar} chof - "$$tardir"
am__untar = $${TAR-tar} xf -
bindir = ${exec_prefix}/bin
build = x86_64-unknown-linux-gnu
build_alias = 
build_cpu = x86_64
build_os = linux-gnu
build_vendor = unknown
builddir = .
datadir = ${datarootdir}
datarootdir = ${prefix}/share
docdir = ${datarootdir}/doc/${PACKAGE_TARNAME}
dvidir = ${docdir}
exec_prefix = ${prefix}
host = x86_64-unknown-linux-gnu
host_alias = 
host_cpu = x86_64
host_os = linux-gnu
host_vendor = unknown
htmldir = ${docdir}
includedir = ${prefix}/include
infodir = ${datarootdir}/info
install_sh = ${SHELL} /home/yonif/A_PAZI/ESR-CG/mpi-pmem-ext-harel/mpi_one_sided_extension/install-sh
libdir = ${exec_prefix}/lib
libexecdir = ${exec_prefix}/libexec
localedir = ${datarootdir}/locale
localstatedir = ${prefix}/var
mandir = ${datarootdir}/man
mkdir_p = $(MKDIR_P)
oldincludedir = /usr/include
pdfdir = ${docdir}
prefix = /home/yonif/installed-mpi-pmem/mpi-win-pm-ext/mpich-gnu/2016
program_transform_name = s,x,x,
psdir = ${docdir}
sbindir = ${exec_prefix}/sbin
sharedstatedir = ${prefix}/com
srcdir = .
sysconfdir = ${prefix}/etc
target_alias = 
top_build_prefix = ../
top_builddir = ..
top_srcdir = ..
AUTOMAKE_OPTIONS = subdir-objects
mpi_one_sided_SOURCES = main.c
HPCCG_HPCCG_SOURCES = HPCCG/compute_residual.cpp HPCCG/compute_residual.hpp HPCCG/ddot.cpp HPCCG/ddot.hpp HPCCG/dump_matlab_matrix.cpp HPCCG/dump_matlab_matrix.hpp \
                  HPCCG/exchange_externals.cpp HPCCG/exchange_externals.hpp HPCCG/generate_matrix.cpp HPCCG/generate_matrix.hpp HPCCG/HPC_Sparse_Matrix.cpp HPCCG/HPC_Sparse_Matrix.hpp \
                  HPCCG/HPC_sparsemv.cpp HPCCG/HPC_sparsemv.hpp HPCCG/HPCCG.cpp HPCCG/HPCCG.hpp HPCCG/make_local_matrix.cpp HPCCG/make_local_matrix.hpp HPCCG/mytimer.cpp HPCCG/mytimer.hpp \
                  HPCCG/read_HPC_row.cpp HPCCG/read_HPC_row.hpp HPCCG/waxpby.cpp HPCCG/waxpby.hpp HPCCG/YAML_Doc.cpp HPCCG/YAML_Doc.hpp HPCCG/YAML_Element.cpp HPCCG/YAML_Element.hpp \
                  HPCCG/HPCCG_main.cpp

HPCCG_HPCCG_checkpoint_disk_SOURCES = HPCCG/compute_residual.cpp HPCCG/compute_residual.hpp HPCCG/ddot.cpp HPCCG/ddot.hpp HPCCG/dump_matlab_matrix.cpp HPCCG/dump_matlab_matrix.hpp \
                                HPCCG/exchange_externals.cpp HPCCG/exchange_externals.hpp HPCCG/generate_matrix.cpp HPCCG/generate_matrix.hpp HPCCG/HPC_Sparse_Matrix.cpp HPCCG/HPC_Sparse_Matrix.hpp \
                                HPCCG/HPC_sparsemv.cpp HPCCG/HPC_sparsemv.hpp HPCCG/HPCCG_checkpoint_disk.cpp HPCCG/HPCCG.hpp HPCCG/make_local_matrix.cpp HPCCG/make_local_matrix.hpp \
                                HPCCG/mytimer.cpp HPCCG/mytimer.hpp HPCCG/read_HPC_row.cpp HPCCG/read_HPC_row.hpp HPCCG/waxpby.cpp HPCCG/waxpby.hpp HPCCG/YAML_Doc.cpp HPCCG/YAML_Doc.hpp \
                                HPCCG/YAML_Element.cpp HPCCG/YAML_Element.hpp HPCCG/HPCCG_main.cpp

HPCCG_HPCCG_checkpoint_pmem_SOURCES = HPCCG/compute_residual.cpp HPCCG/compute_residual.hpp HPCCG/ddot.cpp HPCCG/ddot.hpp HPCCG/dump_matlab_matrix.cpp HPCCG/dump_matlab_matrix.hpp \
                                HPCCG/exchange_externals.cpp HPCCG/exchange_externals.hpp HPCCG/generate_matrix.cpp HPCCG/generate_matrix.hpp HPCCG/HPC_Sparse_Matrix.cpp HPCCG/HPC_Sparse_Matrix.hpp \
                                HPCCG/HPC_sparsemv.cpp HPCCG/HPC_sparsemv.hpp HPCCG/HPCCG_checkpoint_pmem.cpp HPCCG/HPCCG.hpp HPCCG/make_local_matrix.cpp HPCCG/make_local_matrix.hpp \
                                HPCCG/mytimer.cpp HPCCG/mytimer.hpp HPCCG/read_HPC_row.cpp HPCCG/read_HPC_row.hpp HPCCG/waxpby.cpp HPCCG/waxpby.hpp HPCCG/YAML_Doc.cpp HPCCG/YAML_Doc.hpp \
                                HPCCG/YAML_Element.cpp HPCCG/YAML_Element.hpp HPCCG/HPCCG_main.cpp

HPCCG_HPCCG_checkpoint_double_buffered_SOURCES = HPCCG/compute_residual.cpp HPCCG/compute_residual.hpp HPCCG/ddot.cpp HPCCG/ddot.hpp HPCCG/dump_matlab_matrix.cpp HPCCG/dump_matlab_matrix.hpp \
                                            HPCCG/exchange_externals.cpp HPCCG/exchange_externals.hpp HPCCG/generate_matrix.cpp HPCCG/generate_matrix.hpp \
                                            HPCCG/HPC_Sparse_Matrix.cpp HPCCG/HPC_Sparse_Matrix.hpp HPCCG/HPC_sparsemv.cpp HPCCG/HPC_sparsemv.hpp HPCCG/HPCCG_checkpoint_double_buffered.cpp \
                                            HPCCG/HPCCG.hpp HPCCG/make_local_matrix.cpp HPCCG/make_local_matrix.hpp HPCCG/mytimer.cpp HPCCG/mytimer.hpp HPCCG/read_HPC_row.cpp HPCCG/read_HPC_row.hpp \
                                            HPCCG/waxpby.cpp HPCCG/waxpby.hpp HPCCG/YAML_Doc.cpp HPCCG/YAML_Doc.hpp HPCCG/YAML_Element.cpp HPCCG/YAML_Element.hpp HPCCG/HPCCG_main.cpp

all: all-am

.SUFFIXES:
.SUFFIXES: .c .cpp .lo .o .obj
$(srcdir)/Makefile.in:  $(srcdir)/Makefile.am  $(am__configure_deps)
	@for dep in $?; do \
	  case '$(am__configure_deps)' in \
	    *$$dep*) \
	      ( cd $(top_builddir) && $(MAKE) $(AM_MAKEFLAGS) am--refresh ) \
	        && { if test -f $@; then exit 0; else break; fi; }; \
	      exit 1;; \
	  esac; \
	done; \
	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign examples/Makefile'; \
	$(am__cd) $(top_srcdir) && \
	  $(AUTOMAKE) --foreign examples/Makefile
Makefile: $(srcdir)/Makefile.in $(top_builddir)/config.status
	@case '$?' in \
	  *config.status*) \
	    cd $(top_builddir) && $(MAKE) $(AM_MAKEFLAGS) am--refresh;; \
	  *) \
	    echo ' cd $(top_builddir) && $(SHELL) ./config.status $(subdir)/$@ $(am__maybe_remake_depfiles)'; \
	    cd $(top_builddir) && $(SHELL) ./config.status $(subdir)/$@ $(am__maybe_remake_depfiles);; \
	esac;

$(top_builddir)/config.status: $(top_srcdir)/configure $(CONFIG_STATUS_DEPENDENCIES)
	cd $(top_builddir) && $(MAKE) $(AM_MAKEFLAGS) am--refresh

$(top_srcdir)/configure:  $(am__configure_deps)
	cd $(top_builddir) && $(MAKE) $(AM_MAKEFLAGS) am--refresh
$(ACLOCAL_M4):  $(am__aclocal_m4_deps)
	cd $(top_builddir) && $(MAKE) $(AM_MAKEFLAGS) am--refresh
$(am__aclocal_m4_deps):
install-binPROGRAMS: $(bin_PROGRAMS)
	@$(NORMAL_INSTALL)
	@list='$(bin_PROGRAMS)'; test -n "$(bindir)" || list=; \
	if test -n "$$list"; then \
	  echo " $(MKDIR_P) '$(DESTDIR)$(bindir)'"; \
	  $(MKDIR_P) "$(DESTDIR)$(bindir)" || exit 1; \
	fi; \
	for p in $$list; do echo "$$p $$p"; done | \
	sed 's/$(EXEEXT)$$//' | \
	while read p p1; do if test -f $$p \
	 || test -f $$p1 \
	  ; then echo "$$p"; echo "$$p"; else :; fi; \
	done | \
	sed -e 'p;s,.*/,,;n;h' \
	    -e 's|.*|.|' \
	    -e 'p;x;s,.*/,,;s/$(EXEEXT)$$//;$(transform);s/$$/$(EXEEXT)/' | \
	sed 'N;N;N;s,\n, ,g' | \
	$(AWK) 'BEGIN { files["."] = ""; dirs["."] = 1 } \
	  { d=$$3; if (dirs[d] != 1) { print "d", d; dirs[d] = 1 } \
	    if ($$2 == $$4) files[d] = files[d] " " $$1; \
	    else { print "f", $$3 "/" $$4, $$1; } } \
	  END { for (d in files) print "f", d, files[d] }' | \
	while read type dir files; do \
	    if test "$$dir" = .; then dir=; else dir=/$$dir; fi; \
	    test -z "$$files" || { \
	    echo " $(INSTALL_PROGRAM_ENV) $(LIBTOOL) $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=install $(INSTALL_PROGRAM) $$files '$(DESTDIR)$(bindir)$$dir'"; \
	    $(INSTALL_PROGRAM_ENV) $(LIBTOOL) $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=install $(INSTALL_PROGRAM) $$files "$(DESTDIR)$(bindir)$$dir" || exit $$?; \
	    } \
	; done

uninstall-binPROGRAMS:
	@$(NORMAL_UNINSTALL)
	@list='$(bin_PROGRAMS)'; test -n "$(bindir)" || list=; \
	files=`for p in $$list; do echo "$$p"; done | \
	  sed -e 'h;s,^.*/,,;s/$(EXEEXT)$$//;$(transform)' \
	      -e 's/$$/$(EXEEXT)/' \
	`; \
	test -n "$$list" || exit 0; \
	echo " ( cd '$(DESTDIR)$(bindir)' && rm -f" $$files ")"; \
	cd "$(DESTDIR)$(bindir)" && rm -f $$files

clean-binPROGRAMS:
	@list='$(bin_PROGRAMS)'; test -n "$$list" || exit 0; \
	echo " rm -f" $$list; \
	rm -f $$list || exit $$?; \
	test -n "$(EXEEXT)" || exit 0; \
	list=`for p in $$list; do echo "$$p"; done | sed 's/$(EXEEXT)$$//'`; \
	echo " rm -f" $$list; \
	rm -f $$list
HPCCG/$(am__dirstamp):
	@$(MKDIR_P) HPCCG
	@: > HPCCG/$(am__dirstamp)
HPCCG/$(DEPDIR)/$(am__dirstamp):
	@$(MKDIR_P) HPCCG/$(DEPDIR)
	@: > HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/compute_residual.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/ddot.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/dump_matlab_matrix.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/exchange_externals.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/generate_matrix.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/HPC_Sparse_Matrix.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/HPC_sparsemv.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/HPCCG.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/make_local_matrix.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/mytimer.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/read_HPC_row.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/waxpby.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/YAML_Doc.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/YAML_Element.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)
HPCCG/HPCCG_main.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)

HPCCG/HPCCG$(EXEEXT): $(HPCCG_HPCCG_OBJECTS) $(HPCCG_HPCCG_DEPENDENCIES) $(EXTRA_HPCCG_HPCCG_DEPENDENCIES) HPCCG/$(am__dirstamp)
	@rm -f HPCCG/HPCCG$(EXEEXT)
	$(AM_V_CXXLD)$(CXXLINK) $(HPCCG_HPCCG_OBJECTS) $(HPCCG_HPCCG_LDADD) $(LIBS)
HPCCG/HPCCG_checkpoint_disk.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)

HPCCG/HPCCG_checkpoint_disk$(EXEEXT): $(HPCCG_HPCCG_checkpoint_disk_OBJECTS) $(HPCCG_HPCCG_checkpoint_disk_DEPENDENCIES) $(EXTRA_HPCCG_HPCCG_checkpoint_disk_DEPENDENCIES) HPCCG/$(am__dirstamp)
	@rm -f HPCCG/HPCCG_checkpoint_disk$(EXEEXT)
	$(AM_V_CXXLD)$(CXXLINK) $(HPCCG_HPCCG_checkpoint_disk_OBJECTS) $(HPCCG_HPCCG_checkpoint_disk_LDADD) $(LIBS)
HPCCG/HPCCG_checkpoint_double_buffered.$(OBJEXT):  \
	HPCCG/$(am__dirstamp) HPCCG/$(DEPDIR)/$(am__dirstamp)

HPCCG/HPCCG_checkpoint_double_buffered$(EXEEXT): $(HPCCG_HPCCG_checkpoint_double_buffered_OBJECTS) $(HPCCG_HPCCG_checkpoint_double_buffered_DEPENDENCIES) $(EXTRA_HPCCG_HPCCG_checkpoint_double_buffered_DEPENDENCIES) HPCCG/$(am__dirstamp)
	@rm -f HPCCG/HPCCG_checkpoint_double_buffered$(EXEEXT)
	$(AM_V_CXXLD)$(CXXLINK) $(HPCCG_HPCCG_checkpoint_double_buffered_OBJECTS) $(HPCCG_HPCCG_checkpoint_double_buffered_LDADD) $(LIBS)
HPCCG/HPCCG_checkpoint_pmem.$(OBJEXT): HPCCG/$(am__dirstamp) \
	HPCCG/$(DEPDIR)/$(am__dirstamp)

HPCCG/HPCCG_checkpoint_pmem$(EXEEXT): $(HPCCG_HPCCG_checkpoint_pmem_OBJECTS) $(HPCCG_HPCCG_checkpoint_pmem_DEPENDENCIES) $(EXTRA_HPCCG_HPCCG_checkpoint_pmem_DEPENDENCIES) HPCCG/$(am__dirstamp)
	@rm -f HPCCG/HPCCG_checkpoint_pmem$(EXEEXT)
	$(AM_V_CXXLD)$(CXXLINK) $(HPCCG_HPCCG_checkpoint_pmem_OBJECTS) $(HPCCG_HPCCG_checkpoint_pmem_LDADD) $(LIBS)

mpi_one_sided$(EXEEXT): $(mpi_one_sided_OBJECTS) $(mpi_one_sided_DEPENDENCIES) $(EXTRA_mpi_one_sided_DEPENDENCIES) 
	@rm -f mpi_one_sided$(EXEEXT)
	$(AM_V_CCLD)$(LINK) $(mpi_one_sided_OBJECTS) $(mpi_one_sided_LDADD) $(LIBS)

mostlyclean-compile:
	-rm -f *.$(OBJEXT)
	-rm -f HPCCG/*.$(OBJEXT)

distclean-compile:
	-rm -f *.tab.c

include ./$(DEPDIR)/main.Po # am--include-marker
include HPCCG/$(DEPDIR)/HPCCG.Po # am--include-marker
include HPCCG/$(DEPDIR)/HPCCG_checkpoint_disk.Po # am--include-marker
include HPCCG/$(DEPDIR)/HPCCG_checkpoint_double_buffered.Po # am--include-marker
include HPCCG/$(DEPDIR)/HPCCG_checkpoint_pmem.Po # am--include-marker
include HPCCG/$(DEPDIR)/HPCCG_main.Po # am--include-marker
include HPCCG/$(DEPDIR)/HPC_Sparse_Matrix.Po # am--include-marker
include HPCCG/$(DEPDIR)/HPC_sparsemv.Po # am--include-marker
include HPCCG/$(DEPDIR)/YAML_Doc.Po # am--include-marker
include HPCCG/$(DEPDIR)/YAML_Element.Po # am--include-marker
include HPCCG/$(DEPDIR)/compute_residual.Po # am--include-marker
include HPCCG/$(DEPDIR)/ddot.Po # am--include-marker
include HPCCG/$(DEPDIR)/dump_matlab_matrix.Po # am--include-marker
include HPCCG/$(DEPDIR)/exchange_externals.Po # am--include-marker
include HPCCG/$(DEPDIR)/generate_matrix.Po # am--include-marker
include HPCCG/$(DEPDIR)/make_local_matrix.Po # am--include-marker
include HPCCG/$(DEPDIR)/mytimer.Po # am--include-marker
include HPCCG/$(DEPDIR)/read_HPC_row.Po # am--include-marker
include HPCCG/$(DEPDIR)/waxpby.Po # am--include-marker

$(am__depfiles_remade):
	@$(MKDIR_P) $(@D)
	@echo '# dummy' >$@-t && $(am__mv) $@-t $@

am--depfiles: $(am__depfiles_remade)

.c.o:
	$(AM_V_CC)depbase=`echo $@ | sed 's|[^/]*$$|$(DEPDIR)/&|;s|\.o$$||'`;\
	$(COMPILE) -MT $@ -MD -MP -MF $$depbase.Tpo -c -o $@ $< &&\
	$(am__mv) $$depbase.Tpo $$depbase.Po
#	$(AM_V_CC)source='$<' object='$@' libtool=no \
#	DEPDIR=$(DEPDIR) $(CCDEPMODE) $(depcomp) \
#	$(AM_V_CC_no)$(COMPILE) -c -o $@ $<

.c.obj:
	$(AM_V_CC)depbase=`echo $@ | sed 's|[^/]*$$|$(DEPDIR)/&|;s|\.obj$$||'`;\
	$(COMPILE) -MT $@ -MD -MP -MF $$depbase.Tpo -c -o $@ `$(CYGPATH_W) '$<'` &&\
	$(am__mv) $$depbase.Tpo $$depbase.Po
#	$(AM_V_CC)source='$<' object='$@' libtool=no \
#	DEPDIR=$(DEPDIR) $(CCDEPMODE) $(depcomp) \
#	$(AM_V_CC_no)$(COMPILE) -c -o $@ `$(CYGPATH_W) '$<'`

.c.lo:
	$(AM_V_CC)depbase=`echo $@ | sed 's|[^/]*$$|$(DEPDIR)/&|;s|\.lo$$||'`;\
	$(LTCOMPILE) -MT $@ -MD -MP -MF $$depbase.Tpo -c -o $@ $< &&\
	$(am__mv) $$depbase.Tpo $$depbase.Plo
#	$(AM_V_CC)source='$<' object='$@' libtool=yes \
#	DEPDIR=$(DEPDIR) $(CCDEPMODE) $(depcomp) \
#	$(AM_V_CC_no)$(LTCOMPILE) -c -o $@ $<

.cpp.o:
	$(AM_V_CXX)depbase=`echo $@ | sed 's|[^/]*$$|$(DEPDIR)/&|;s|\.o$$||'`;\
	$(CXXCOMPILE) -MT $@ -MD -MP -MF $$depbase.Tpo -c -o $@ $< &&\
	$(am__mv) $$depbase.Tpo $$depbase.Po
#	$(AM_V_CXX)source='$<' object='$@' libtool=no \
#	DEPDIR=$(DEPDIR) $(CXXDEPMODE) $(depcomp) \
#	$(AM_V_CXX_no)$(CXXCOMPILE) -c -o $@ $<

.cpp.obj:
	$(AM_V_CXX)depbase=`echo $@ | sed 's|[^/]*$$|$(DEPDIR)/&|;s|\.obj$$||'`;\
	$(CXXCOMPILE) -MT $@ -MD -MP -MF $$depbase.Tpo -c -o $@ `$(CYGPATH_W) '$<'` &&\
	$(am__mv) $$depbase.Tpo $$depbase.Po
#	$(AM_V_CXX)source='$<' object='$@' libtool=no \
#	DEPDIR=$(DEPDIR) $(CXXDEPMODE) $(depcomp) \
#	$(AM_V_CXX_no)$(CXXCOMPILE) -c -o $@ `$(CYGPATH_W) '$<'`

.cpp.lo:
	$(AM_V_CXX)depbase=`echo $@ | sed 's|[^/]*$$|$(DEPDIR)/&|;s|\.lo$$||'`;\
	$(LTCXXCOMPILE) -MT $@ -MD -MP -MF $$depbase.Tpo -c -o $@ $< &&\
	$(am__mv) $$depbase.Tpo $$depbase.Plo
#	$(AM_V_CXX)source='$<' object='$@' libtool=yes \
#	DEPDIR=$(DEPDIR) $(CXXDEPMODE) $(depcomp) \
#	$(AM_V_CXX_no)$(LTCXXCOMPILE) -c -o $@ $<

mostlyclean-libtool:
	-rm -f *.lo

clean-libtool:
	-rm -rf .libs _libs
	-rm -rf HPCCG/.libs HPCCG/_libs

ID: $(am__tagged_files)
	$(am__define_uniq_tagged_files); mkid -fID $$unique
tags: tags-am
TAGS: tags

tags-am: $(TAGS_DEPENDENCIES) $(am__tagged_files)
	set x; \
	here=`pwd`; \
	$(am__define_uniq_tagged_files); \
	shift; \
	if test -z "$(ETAGS_ARGS)$$*$$unique"; then :; else \
	  test -n "$$unique" || unique=$$empty_fix; \
	  if test $$# -gt 0; then \
	    $(ETAGS) $(ETAGSFLAGS) $(AM_ETAGSFLAGS) $(ETAGS_ARGS) \
	      "$$@" $$unique; \
	  else \
	    $(ETAGS) $(ETAGSFLAGS) $(AM_ETAGSFLAGS) $(ETAGS_ARGS) \
	      $$unique; \
	  fi; \
	fi
ctags: ctags-am

CTAGS: ctags
ctags-am: $(TAGS_DEPENDENCIES) $(am__tagged_files)
	$(am__define_uniq_tagged_files); \
	test -z "$(CTAGS_ARGS)$$unique" \
	  || $(CTAGS) $(CTAGSFLAGS) $(AM_CTAGSFLAGS) $(CTAGS_ARGS) \
	     $$unique

GTAGS:
	here=`$(am__cd) $(top_builddir) && pwd` \
	  && $(am__cd) $(top_srcdir) \
	  && gtags -i $(GTAGS_ARGS) "$$here"
cscopelist: cscopelist-am

cscopelist-am: $(am__tagged_files)
	list='$(am__tagged_files)'; \
	case "$(srcdir)" in \
	  [\\/]* | ?:[\\/]*) sdir="$(srcdir)" ;; \
	  *) sdir=$(subdir)/$(srcdir) ;; \
	esac; \
	for i in $$list; do \
	  if test -f "$$i"; then \
	    echo "$(subdir)/$$i"; \
	  else \
	    echo "$$sdir/$$i"; \
	  fi; \
	done >> $(top_builddir)/cscope.files

distclean-tags:
	-rm -f TAGS ID GTAGS GRTAGS GSYMS GPATH tags
distdir: $(BUILT_SOURCES)
	$(MAKE) $(AM_MAKEFLAGS) distdir-am

distdir-am: $(DISTFILES)
	@srcdirstrip=`echo "$(srcdir)" | sed 's/[].[^$$\\*]/\\\\&/g'`; \
	topsrcdirstrip=`echo "$(top_srcdir)" | sed 's/[].[^$$\\*]/\\\\&/g'`; \
	list='$(DISTFILES)'; \
	  dist_files=`for file in $$list; do echo $$file; done | \
	  sed -e "s|^$$srcdirstrip/||;t" \
	      -e "s|^$$topsrcdirstrip/|$(top_builddir)/|;t"`; \
	case $$dist_files in \
	  */*) $(MKDIR_P) `echo "$$dist_files" | \
			   sed '/\//!d;s|^|$(distdir)/|;s,/[^/]*$$,,' | \
			   sort -u` ;; \
	esac; \
	for file in $$dist_files; do \
	  if test -f $$file || test -d $$file; then d=.; else d=$(srcdir); fi; \
	  if test -d $$d/$$file; then \
	    dir=`echo "/$$file" | sed -e 's,/[^/]*$$,,'`; \
	    if test -d "$(distdir)/$$file"; then \
	      find "$(distdir)/$$file" -type d ! -perm -700 -exec chmod u+rwx {} \;; \
	    fi; \
	    if test -d $(srcdir)/$$file && test $$d != $(srcdir); then \
	      cp -fpR $(srcdir)/$$file "$(distdir)$$dir" || exit 1; \
	      find "$(distdir)/$$file" -type d ! -perm -700 -exec chmod u+rwx {} \;; \
	    fi; \
	    cp -fpR $$d/$$file "$(distdir)$$dir" || exit 1; \
	  else \
	    test -f "$(distdir)/$$file" \
	    || cp -p $$d/$$file "$(distdir)/$$file" \
	    || exit 1; \
	  fi; \
	done
check-am: all-am
check: check-am
all-am: Makefile $(PROGRAMS)
installdirs:
	for dir in "$(DESTDIR)$(bindir)"; do \
	  test -z "$$dir" || $(MKDIR_P) "$$dir"; \
	done
install: install-am
install-exec: install-exec-am
install-data: install-data-am
uninstall: uninstall-am

install-am: all-am
	@$(MAKE) $(AM_MAKEFLAGS) install-exec-am install-data-am

installcheck: installcheck-am
install-strip:
	if test -z '$(STRIP)'; then \
	  $(MAKE) $(AM_MAKEFLAGS) INSTALL_PROGRAM="$(INSTALL_STRIP_PROGRAM)" \
	    install_sh_PROGRAM="$(INSTALL_STRIP_PROGRAM)" INSTALL_STRIP_FLAG=-s \
	      install; \
	else \
	  $(MAKE) $(AM_MAKEFLAGS) INSTALL_PROGRAM="$(INSTALL_STRIP_PROGRAM)" \
	    install_sh_PROGRAM="$(INSTALL_STRIP_PROGRAM)" INSTALL_STRIP_FLAG=-s \
	    "INSTALL_PROGRAM_ENV=STRIPPROG='$(STRIP)'" install; \
	fi
mostlyclean-generic:

clean-generic:

distclean-generic:
	-test -z "$(CONFIG_CLEAN_FILES)" || rm -f $(CONFIG_CLEAN_FILES)
	-test . = "$(srcdir)" || test -z "$(CONFIG_CLEAN_VPATH_FILES)" || rm -f $(CONFIG_CLEAN_VPATH_FILES)
	-rm -f HPCCG/$(DEPDIR)/$(am__dirstamp)
	-rm -f HPCCG/$(am__dirstamp)

maintainer-clean-generic:
	@echo "This command is intended for maintainers to use"
	@echo "it deletes files that may require special tools to rebuild."
clean: clean-am

clean-am: clean-binPROGRAMS clean-generic clean-libtool mostlyclean-am

distclean: distclean-am
		-rm -f ./$(DEPDIR)/main.Po
	-rm -f HPCCG/$(DEPDIR)/HPCCG.Po
	-rm -f HPCCG/$(DEPDIR)/HPCCG_checkpoint_disk.Po
	-rm -f HPCCG/$(DEPDIR)/HPCCG_checkpoint_double_buffered.Po
	-rm -f HPCCG/$(DEPDIR)/HPCCG_checkpoint_pmem.Po
	-rm -f HPCCG/$(DEPDIR)/HPCCG_main.Po
	-rm -f HPCCG/$(DEPDIR)/HPC_Sparse_Matrix.Po
	-rm -f HPCCG/$(DEPDIR)/HPC_sparsemv.Po
	-rm -f HPCCG/$(DEPDIR)/YAML_Doc.Po
	-rm -f HPCCG/$(DEPDIR)/YAML_Element.Po
	-rm -f HPCCG/$(DEPDIR)/compute_residual.Po
	-rm -f HPCCG/$(DEPDIR)/ddot.Po
	-rm -f HPCCG/$(DEPDIR)/dump_matlab_matrix.Po
	-rm -f HPCCG/$(DEPDIR)/exchange_externals.Po
	-rm -f HPCCG/$(DEPDIR)/generate_matrix.Po
	-rm -f HPCCG/$(DEPDIR)/make_local_matrix.Po
	-rm -f HPCCG/$(DEPDIR)/mytimer.Po
	-rm -f HPCCG/$(DEPDIR)/read_HPC_row.Po
	-rm -f HPCCG/$(DEPDIR)/waxpby.Po
	-rm -f Makefile
distclean-am: clean-am distclean-compile distclean-generic \
	distclean-tags

dvi: dvi-am

dvi-am:

html: html-am

html-am:

info: info-am

info-am:

install-data-am:

install-dvi: install-dvi-am

install-dvi-am:

install-exec-am: install-binPROGRAMS

install-html: install-html-am

install-html-am:

install-info: install-info-am

install-info-am:

install-man:

install-pdf: install-pdf-am

install-pdf-am:

install-ps: install-ps-am

install-ps-am:

installcheck-am:

maintainer-clean: maintainer-clean-am
		-rm -f ./$(DEPDIR)/main.Po
	-rm -f HPCCG/$(DEPDIR)/HPCCG.Po
	-rm -f HPCCG/$(DEPDIR)/HPCCG_checkpoint_disk.Po
	-rm -f HPCCG/$(DEPDIR)/HPCCG_checkpoint_double_buffered.Po
	-rm -f HPCCG/$(DEPDIR)/HPCCG_checkpoint_pmem.Po
	-rm -f HPCCG/$(DEPDIR)/HPCCG_main.Po
	-rm -f HPCCG/$(DEPDIR)/HPC_Sparse_Matrix.Po
	-rm -f HPCCG/$(DEPDIR)/HPC_sparsemv.Po
	-rm -f HPCCG/$(DEPDIR)/YAML_Doc.Po
	-rm -f HPCCG/$(DEPDIR)/YAML_Element.Po
	-rm -f HPCCG/$(DEPDIR)/compute_residual.Po
	-rm -f HPCCG/$(DEPDIR)/ddot.Po
	-rm -f HPCCG/$(DEPDIR)/dump_matlab_matrix.Po
	-rm -f HPCCG/$(DEPDIR)/exchange_externals.Po
	-rm -f HPCCG/$(DEPDIR)/generate_matrix.Po
	-rm -f HPCCG/$(DEPDIR)/make_local_matrix.Po
	-rm -f HPCCG/$(DEPDIR)/mytimer.Po
	-rm -f HPCCG/$(DEPDIR)/read_HPC_row.Po
	-rm -f HPCCG/$(DEPDIR)/waxpby.Po
	-rm -f Makefile
maintainer-clean-am: distclean-am maintainer-clean-generic

mostlyclean: mostlyclean-am

mostlyclean-am: mostlyclean-compile mostlyclean-generic \
	mostlyclean-libtool

pdf: pdf-am

pdf-am:

ps: ps-am

ps-am:

uninstall-am: uninstall-binPROGRAMS

.MAKE: install-am install-strip

.PHONY: CTAGS GTAGS TAGS all all-am am--depfiles check check-am clean \
	clean-binPROGRAMS clean-generic clean-libtool cscopelist-am \
	ctags ctags-am distclean distclean-compile distclean-generic \
	distclean-libtool distclean-tags distdir dvi dvi-am html \
	html-am info info-am install install-am install-binPROGRAMS \
	install-data install-data-am install-dvi install-dvi-am \
	install-exec install-exec-am install-html install-html-am \
	install-info install-info-am install-man install-pdf \
	install-pdf-am install-ps install-ps-am install-strip \
	installcheck installcheck-am installdirs maintainer-clean \
	maintainer-clean-generic mostlyclean mostlyclean-compile \
	mostlyclean-generic mostlyclean-libtool pdf pdf-am ps ps-am \
	tags tags-am uninstall uninstall-am uninstall-binPROGRAMS

.PRECIOUS: Makefile


# Tell versions [3.59,3.63) of GNU make to not export all variables.
# Otherwise a system limit (for SysV at least) may be exceeded.
.NOEXPORT:
